import numpy as np
import torch

from connpre import neural
from connpre.data import synthetic_cohort
from connpre.finetune import FinetuneConfig, finetune, predict, random_state
from connpre.model import ModelConfig
from connpre.neural import (
    LR_GRIDS,
    calibration_without_tuning,
    fold_neural,
    fold_seed,
    lr_selection_sets,
    probe_arm,
    select_lrs,
)
from connpre.pretrain import PretrainConfig, eval_starts
from connpre.protocol import N_ALL, draw_sets, loso_folds
from connpre.results import RunStore

SMALL = ModelConfig(n_roi=12, window=24, dim=32, depth=2, heads=4, mlp=64)
PRE = PretrainConfig(steps=20, batch=8, log_every=10)
N_CAL, N_LR = 40, 20


def synth():
    cohort = synthetic_cohort(sizes=(60, 50, 40, 40, 30), n_roi=12, t=64, seed=5)
    return cohort, torch.as_tensor(cohort.x), loso_folds(cohort.site, 30)[0]


def run(x, cohort, fold, trans=None):
    return fold_neural(
        cohort,
        x,
        fold,
        "loso",
        seeds=(0,),
        sizes=(30, 60),
        n_cal=N_CAL,
        pre_cfg=PRE,
        model_cfg=SMALL,
        trans_state=trans,
        ft_steps={30: 4, 60: 4},
        lr_steps=4,
        lr_n_lab=N_LR,
        log=lambda *_: None,
    )


def test_probe_arm_names():
    assert probe_arm("mean", "pre") == "probe_mean"
    assert probe_arm("flat", "rand") == "probe_flat_rand"
    assert probe_arm("mean", "trans") == "probe_mean_trans"


def test_fold_runs_every_arm_for_both_targets_and_sizes():
    cohort, x, fold = synth()
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
        "ft_trans",
        "ft_pre_trans",
    }
    assert len(store.rows) == 10 * 2 * 2
    allowed = {
        "ft_pre": LR_GRIDS["ft_pre"],
        "ft_scratch": LR_GRIDS["ft_scratch"],
        "ft_trans": LR_GRIDS["ft_trans"],
        "ft_pre_trans": LR_GRIDS["ft_pre"],
    }
    assert all(r["lr"] in allowed[r["arm"]] for r in store.rows if r["arm"] in allowed)
    assert store.extra["pretrain"]["history"][-1]["step"] == 20
    assert set(store.extra["lr"]) == {
        f"{a}|{t}" for a in ("ft_pre", "ft_scratch", "ft_trans") for t in ("age", "fiq")
    }
    assert store.extra["lr_subjects"]["labelled"] == N_LR


def test_without_a_transductive_encoder_only_the_inductive_arms_run():
    cohort, x, fold = synth()
    store = run(x, cohort, fold)
    arms = {r["arm"] for r in store.rows}
    assert arms == {a for a in arms if "trans" not in a}
    assert len(arms) == 6
    assert {k.split("|")[0] for k in store.extra["lr"]} == {"ft_pre", "ft_scratch"}


def test_held_out_series_are_never_used_to_train_or_calibrate():
    cohort, x, fold = synth()
    x = x.clone()
    x[fold.test] = float("nan")
    store = run(x, cohort, fold)
    history = store.extra["pretrain"]["history"]
    assert all(np.isfinite(h["train_loss"]) for h in history)
    assert all(np.isfinite(h["val_ratio"]) for h in history)
    assert all(np.isnan(h["heldout_ratio"]) for h in history)
    assert all(np.isfinite(r["q90"]) for r in store.rows)
    assert all(np.isfinite(v) for v in store.extra["lr"].values())


def test_rate_subjects_come_from_the_training_set_outside_the_validation_site():
    cohort, _, fold = synth()
    site = cohort.site
    lab, val = lr_selection_sets(fold.train, site, fold.name, True, 3, N_LR)
    assert len(lab) == N_LR and len(np.unique(lab)) == N_LR
    assert set(lab) | set(val) <= set(fold.train)
    assert not set(lab) & set(val)
    assert len(set(site[val])) == 1 and not set(site[lab]) & set(site[val])


def test_finetune_calibration_is_disjoint_from_fit_and_rate_subjects():
    site = np.array(["A"] * 120 + ["B"] * 80 + ["C"] * 60 + ["D"] * 50 + ["E"] * 30)
    dropped_somewhere = False
    for fold in loso_folds(site, 40):
        seed = fold_seed(fold.name)
        lab, val = lr_selection_sets(fold.train, site, fold.name, True, seed, 100)
        tune = set(lab) | set(val)
        for run_seed in range(5):
            draw = draw_sets(fold.train, site, 100, run_seed, fold.name)
            cal = calibration_without_tuning(draw.cal, lab, val)
            dropped_somewhere |= len(cal) < len(draw.cal)
            assert len(cal) > 0 and set(cal) <= set(draw.cal)
            assert not set(cal) & tune
            assert not set(cal) & set(fold.test)
            for n in (50, 100, 300, N_ALL):
                assert not set(cal) & set(draw.labelled(n))
    assert dropped_somewhere


def test_fine_tune_intervals_calibrate_on_subjects_that_chose_no_rate(monkeypatch):
    cohort, x, fold = synth()
    seen = []
    add = RunStore.add

    def spy(self, arm, target, n, seed, pred, cal=None, **extra):
        seen.append((arm, target, cal, extra["n_cal"]))
        return add(self, arm, target, n, seed, pred, cal, **extra)

    monkeypatch.setattr(RunStore, "add", spy)
    run(x, cohort, fold, trans=random_state(SMALL, 7))
    seed = fold_seed(fold.name)
    lab, val = lr_selection_sets(fold.train, cohort.site, fold.name, True, seed, N_LR)
    cal = draw_sets(fold.train, cohort.site, N_CAL, 0, fold.name).cal
    kept = calibration_without_tuning(cal, lab, val)
    assert 0 < len(kept) < len(cal)
    for arm, target, pair, n_cal in seen:
        expected = cal if arm.startswith("probe") else kept
        assert n_cal == len(expected)
        assert np.array_equal(pair[0], cohort.target(target)[expected])


def test_every_arm_is_searched_on_the_same_subjects_for_the_same_budget(monkeypatch):
    cohort, x, fold = synth()
    calls, real = [], neural.finetune

    def spy(state, model_cfg, x, lab, y, cfg, seed):
        calls.append((id(state), tuple(lab), cfg.steps, cfg.lr))
        return real(state, model_cfg, x, lab, y, cfg, seed)

    monkeypatch.setattr(neural, "finetune", spy)
    lab, val = lr_selection_sets(fold.train, cohort.site, fold.name, True, 0, N_LR)
    y_all = np.stack([cohort.age, cohort.fiq], axis=1)
    states = {
        "ft_pre": random_state(SMALL, 1),
        "ft_scratch": None,
        "ft_trans": random_state(SMALL, 2),
    }
    starts = eval_starts(x.shape[1], SMALL.window, 2)
    chosen = select_lrs(x, y_all, lab, val, states, SMALL, starts, 0, steps=3)
    assert set(chosen) == {(arm, t) for arm in states for t in (0, 1)}
    assert LR_GRIDS["ft_trans"] == LR_GRIDS["ft_pre"]
    for arm, state in states.items():
        mine = [c for c in calls if c[0] == id(state)]
        assert len(mine) == 2 * len(LR_GRIDS[arm])
        assert {c[1] for c in mine} == {tuple(lab)}
        assert {c[2] for c in mine} == {3}
        assert {c[3] for c in mine} == set(LR_GRIDS[arm])


def test_select_lrs_picks_the_rate_with_the_lowest_validation_error(monkeypatch):
    cohort = synthetic_cohort(sizes=(30, 30), n_roi=12, t=24, seed=2)
    x = torch.as_tensor(cohort.x)
    signal = cohort.x[:, 0, :].mean(axis=1)
    y_all = np.stack([signal, -signal], axis=1)
    monkeypatch.setitem(LR_GRIDS, "ft_scratch", (1e-9, 3e-3))
    lab, val = np.arange(40), np.arange(40, 60)
    chosen = select_lrs(
        x, y_all, lab, val, {"ft_scratch": None}, SMALL, np.array([0]), 0, steps=120
    )
    assert chosen == {("ft_scratch", 0): 3e-3, ("ft_scratch", 1): 3e-3}
    mu, sd = signal[lab].mean(), signal[lab].std()
    y_val = (signal[val] - mu) / sd
    cfg = FinetuneConfig(lr=chosen[("ft_scratch", 0)], steps=120)
    model = finetune(None, SMALL, x, lab, (signal[lab] - mu) / sd, cfg, 0)
    err = np.mean((predict(model, x, val, np.array([0])) - y_val) ** 2)
    assert err < 0.8 * np.mean(y_val**2)


def test_transductive_start_has_its_own_rate_and_the_old_one_is_an_ablation(
    monkeypatch,
):
    cohort, x, fold = synth()
    monkeypatch.setitem(LR_GRIDS, "ft_pre", (1e-4,))
    monkeypatch.setitem(LR_GRIDS, "ft_trans", (3e-4,))
    store = run(x, cohort, fold, trans=random_state(SMALL, 7))
    assert store.extra["lr"]["ft_trans|age"] == 3e-4
    assert store.extra["lr"]["ft_pre|age"] == 1e-4

    def rates(arm):
        return {r["lr"] for r in store.rows if r["arm"] == arm}

    assert rates("ft_trans") == {3e-4}
    assert rates("ft_pre") == rates("ft_pre_trans") == {1e-4}


def test_each_start_is_searched_from_its_own_weights(monkeypatch):
    cohort, x, fold = synth()
    trans = random_state(SMALL, 7)
    seen, real = [], neural.select_lrs

    def spy(x, y_all, lab, val, states, *args, **kwargs):
        seen.append(states)
        return real(x, y_all, lab, val, states, *args, **kwargs)

    monkeypatch.setattr(neural, "select_lrs", spy)
    run(x, cohort, fold, trans=trans)
    (states,) = seen
    assert set(states) == {"ft_pre", "ft_scratch", "ft_trans"}
    assert states["ft_trans"] is trans
    assert states["ft_scratch"] is None
    assert states["ft_pre"] is not None and states["ft_pre"] is not trans


def test_transductive_arms_share_one_fit_when_their_rates_agree(monkeypatch):
    cohort, x, fold = synth()
    for arm in LR_GRIDS:
        monkeypatch.setitem(LR_GRIDS, arm, (1e-4,))
    calls, real = [], neural.finetune

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(neural, "finetune", spy)
    store = run(x, cohort, fold, trans=random_state(SMALL, 7))
    assert len(calls) == 3 * 2 + 3 * 2 * 2
    rows = {(r["arm"], r["target"], r["n"]): r for r in store.rows}
    for target in ("age", "fiq"):
        for n in (30, 60):
            a, b = rows[("ft_trans", target, n)], rows[("ft_pre_trans", target, n)]
            assert (a["r2"], a["q90"], a["cover90"]) == (
                b["r2"],
                b["q90"],
                b["cover90"],
            )
