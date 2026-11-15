import numpy as np
import torch

from connpre.data import synthetic_cohort
from connpre.finetune import random_state
from connpre.model import ModelConfig
from connpre.neural import LR_GRID, fold_neural, probe_arm
from connpre.pretrain import PretrainConfig
from connpre.protocol import loso_folds

SMALL = ModelConfig(n_roi=12, window=24, dim=32, depth=2, heads=4, mlp=64)
PRE = PretrainConfig(steps=20, batch=8, log_every=10)


def run(x, cohort, fold, trans=None):
    return fold_neural(
        cohort,
        x,
        fold,
        "loso",
        seeds=(0,),
        sizes=(30, 60),
        n_cal=15,
        pre_cfg=PRE,
        model_cfg=SMALL,
        trans_state=trans,
        ft_steps={30: 4, 60: 4},
        lr_steps=4,
        lr_n_lab=30,
        log=lambda *_: None,
    )


def test_probe_arm_names():
    assert probe_arm("mean", "pre") == "probe_mean"
    assert probe_arm("flat", "rand") == "probe_flat_rand"
    assert probe_arm("mean", "trans") == "probe_mean_trans"


def test_fold_runs_every_arm_for_both_targets_and_sizes():
    cohort = synthetic_cohort(sizes=(40, 40, 30, 30, 20), n_roi=12, t=64, seed=5)
    x = torch.as_tensor(cohort.x)
    fold = loso_folds(cohort.site, 30)[0]
    store = run(x, cohort, fold, trans=random_state(SMALL, 7))
    arms = {r["arm"] for r in store.rows}
    assert arms == {
        "probe_mean",
        "probe_flat",
        "probe_mean_rand",
        "probe_flat_rand",
        "probe_mean_trans",
        "probe_flat_trans",
        "ft_pre",
        "ft_scratch",
        "ft_pre_trans",
    }
    assert len(store.rows) == 9 * 2 * 2
    assert all(r["lr"] in LR_GRID for r in store.rows if r["arm"].startswith("ft"))
    assert store.extra["pretrain"]["history"][-1]["step"] == 20
    assert set(store.extra["lr"]) == {
        f"{a}|{t}" for a in ("ft_pre", "ft_scratch") for t in ("age", "fiq")
    }


def test_held_out_series_are_never_used_to_train_or_calibrate():
    cohort = synthetic_cohort(sizes=(40, 40, 30, 30, 20), n_roi=12, t=64, seed=5)
    fold = loso_folds(cohort.site, 30)[0]
    x = torch.as_tensor(cohort.x).clone()
    x[fold.test] = float("nan")
    store = run(x, cohort, fold)
    history = store.extra["pretrain"]["history"]
    assert all(np.isfinite(h["train_loss"]) for h in history)
    assert all(np.isfinite(h["val_ratio"]) for h in history)
    assert all(np.isnan(h["heldout_ratio"]) for h in history)
    assert all(np.isfinite(r["q90"]) for r in store.rows)
    assert all(np.isfinite(v) for v in store.extra["lr"].values())
