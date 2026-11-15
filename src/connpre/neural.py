"""Neural arms on one fold: pretraining, probes and fine-tunes."""

from __future__ import annotations

import time
import zlib

import numpy as np
import torch

from connpre.baselines import new_store
from connpre.data import Cohort
from connpre.finetune import (
    STEPS,
    FinetuneConfig,
    cpu_state,
    embed_all,
    finetune,
    load_encoder,
    predict,
    random_state,
    standardised_gram,
)
from connpre.model import ModelConfig
from connpre.pretrain import PretrainConfig, eval_starts, pretrain
from connpre.protocol import Fold, draw_sets, inner_split, stratified_order
from connpre.ridge import REL_ALPHAS, ridge_cv_predict
from connpre.settings import TARGETS

LR_GRIDS = {
    "ft_pre": (3e-5, 1e-4, 3e-4),
    "ft_scratch": (1e-4, 3e-4, 1e-3),
    "ft_trans": (3e-5, 1e-4, 3e-4),
}
N_MONITOR = 48
RANDOM_SEED = 12345


def probe_arm(kind: str, name: str) -> str:
    return f"probe_{kind}" if name == "pre" else f"probe_{kind}_{name}"


def fold_seed(fold_name: str) -> int:
    return zlib.crc32(fold_name.encode()) % 10_000


def lr_selection_sets(
    train: np.ndarray,
    site: np.ndarray,
    fold_name: str,
    loso: bool,
    seed: int,
    n_lab: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subjects that choose the rates: n_lab labelled ones and the validation set.

    The labelled ones come from the training subjects outside the validation set.
    """
    pool, val = inner_split(train, site, 0, fold_name, loso=loso)
    rng = np.random.default_rng([seed, 99])
    return pool[stratified_order(site[pool], rng)][:n_lab], val


def calibration_without_tuning(
    cal: np.ndarray, lr_lab: np.ndarray, lr_val: np.ndarray
) -> np.ndarray:
    """Calibration subjects that took no part in choosing the learning rates."""
    return cal[~np.isin(cal, np.concatenate([lr_lab, lr_val]))]


def select_lrs(x, y_all, lab, val, states, model_cfg, starts, seed, steps=200) -> dict:
    """Fine-tune learning rate per arm and target, chosen on the validation subjects.

    states maps an arm of LR_GRIDS to its initial weights (None: random weights).
    Every arm trains on the same labelled subjects for the same number of steps and
    searches its own grid.
    """
    chosen = {}
    for arm, state in states.items():
        for t in range(y_all.shape[1]):
            mu, sd = y_all[lab, t].mean(), y_all[lab, t].std()
            y_val = (y_all[val, t] - mu) / sd
            best = None
            for lr in LR_GRIDS[arm]:
                cfg = FinetuneConfig(lr=lr, steps=steps)
                model = finetune(
                    state, model_cfg, x, lab, (y_all[lab, t] - mu) / sd, cfg, seed
                )
                mse = float(np.mean((predict(model, x, val, starts) - y_val) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, lr)
            chosen[(arm, t)] = best[1]
    return chosen


def pretrain_on(x, train_idx, monitor, pre_cfg, model_cfg, seed):
    model, history = pretrain(x, train_idx, pre_cfg, model_cfg, seed, monitor)
    return cpu_state(model), history


def fold_neural(
    cohort: Cohort,
    x: torch.Tensor,
    fold: Fold,
    mode: str,
    seeds: tuple,
    sizes: tuple,
    n_cal: int,
    pre_cfg: PretrainConfig,
    model_cfg: ModelConfig,
    trans_state: dict | None = None,
    ft_steps: dict | None = None,
    lr_steps: int = 200,
    lr_n_lab: int = 100,
    log=print,
):
    """Pretrain on the training subjects of the fold, then run probes and fine-tunes.

    The fine-tune intervals calibrate on the calibration subjects that took no part in
    choosing the learning rates; the probes choose nothing on them and use all of them.
    The transductive encoder is fine-tuned with its own rate (ft_trans) and, as an
    ablation, with the rate of the pretrained start (ft_pre_trans).
    """
    ft_steps = ft_steps or STEPS
    started = time.time()
    site = cohort.site
    y_all = np.stack([cohort.target(t) for t in TARGETS], axis=1)
    fseed = fold_seed(fold.name)
    monitor_idx = np.random.default_rng(fseed).choice(
        fold.train, min(N_MONITOR, len(fold.train) // 5), replace=False
    )
    train_pre = np.setdiff1d(fold.train, monitor_idx)
    monitor = {"val": monitor_idx, "heldout": fold.test}
    state_pre, history = pretrain_on(x, train_pre, monitor, pre_cfg, model_cfg, fseed)
    log(f"{fold.name} pretrained {round(time.time() - started)}s")

    store = new_store(cohort, fold, mode)
    store.extra["pretrain"] = {"history": history, "n_train": int(len(train_pre))}
    starts = eval_starts(x.shape[1], model_cfg.window, 4)
    states = {"pre": state_pre, "rand": random_state(model_cfg, RANDOM_SEED)}
    if trans_state is not None:
        states["trans"] = trans_state
    grams = {}
    for name, state in states.items():
        enc = load_encoder(state, model_cfg, x.device)
        mean_feat, flat_feat = embed_all(enc, x, starts)
        grams[("mean", name)] = standardised_gram(mean_feat, fold.train)
        grams[("flat", name)] = standardised_gram(flat_feat, fold.train)
        del mean_feat, flat_feat

    lr_lab, lr_val = lr_selection_sets(
        fold.train, site, fold.name, mode == "loso", fseed, lr_n_lab
    )
    lr_states = {"ft_pre": state_pre, "ft_scratch": None}
    if trans_state is not None:
        lr_states["ft_trans"] = trans_state
    lrs = select_lrs(
        x, y_all, lr_lab, lr_val, lr_states, model_cfg, starts, fseed, lr_steps
    )
    store.extra["lr"] = {f"{arm}|{TARGETS[t]}": lr for (arm, t), lr in lrs.items()}
    store.extra["lr_subjects"] = {"labelled": len(lr_lab), "validation": len(lr_val)}
    log(f"{fold.name} lr chosen {round(time.time() - started)}s {store.extra['lr']}")

    start_state = {"pre": state_pre, "scratch": None, "trans": trans_state}
    ft_arms = [("ft_pre", "pre", "ft_pre"), ("ft_scratch", "scratch", "ft_scratch")]
    if trans_state is not None:
        ft_arms += [
            ("ft_trans", "trans", "ft_trans"),
            ("ft_pre_trans", "trans", "ft_pre"),
        ]
    for seed in seeds:
        draw = draw_sets(fold.train, site, n_cal, seed, fold.name)
        cal = draw.cal
        cal_ft = calibration_without_tuning(cal, lr_lab, lr_val)
        both, both_ft = (np.concatenate([c, fold.test]) for c in (cal, cal_ft))
        for n in sizes:
            lab = draw.labelled(n)
            for (kind, name), gram in grams.items():
                pred, chosen = ridge_cv_predict(gram, y_all, lab, both, site[lab])
                for t, target in enumerate(TARGETS):
                    cal_pair = (y_all[cal, t], pred[: len(cal), t])
                    store.add(
                        probe_arm(kind, name),
                        target,
                        n,
                        seed,
                        pred[len(cal) :, t],
                        cal_pair,
                        rel_alpha=float(REL_ALPHAS[chosen[t]]),
                        n_cal=len(cal),
                    )
            fits = {}
            for arm, start, lr_arm in ft_arms:
                for t, target in enumerate(TARGETS):
                    lr = lrs[(lr_arm, t)]
                    if (start, t, lr) not in fits:
                        mu, sd = y_all[lab, t].mean(), y_all[lab, t].std()
                        cfg = FinetuneConfig(lr=lr, steps=ft_steps.get(n, 300))
                        ft_seed = 1000 * seed + n % 997 + 17 * t
                        model = finetune(
                            start_state[start],
                            model_cfg,
                            x,
                            lab,
                            (y_all[lab, t] - mu) / sd,
                            cfg,
                            ft_seed,
                        )
                        pred = predict(model, x, both_ft, starts) * sd + mu
                        fits[(start, t, lr)] = pred
                    pred = fits[(start, t, lr)]
                    cal_pair = (y_all[cal_ft, t], pred[: len(cal_ft)])
                    store.add(
                        arm,
                        target,
                        n,
                        seed,
                        pred[len(cal_ft) :],
                        cal_pair,
                        lr=lr,
                        n_cal=len(cal_ft),
                    )
        log(f"{fold.name} seed {seed} done {round(time.time() - started)}s")
    store.extra["seconds"] = round(time.time() - started, 1)
    return store
