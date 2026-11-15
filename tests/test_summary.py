import json

import numpy as np

from connpre.results import RunStore, load_mode
from connpre.summary import clean, fold_means, summarise_mode


def make_store(fold, n_test, offset):
    rng = np.random.default_rng(offset)
    y = {"age": rng.normal(15, 5, n_test), "fiq": rng.normal(105, 12, n_test)}
    store = RunStore(
        "loso",
        fold,
        np.arange(n_test) + 100 * offset,
        np.arange(n_test) + 7 * offset,
        [fold] * n_test,
        y,
    )
    for seed in range(3):
        for target, spread in (("age", 5.0), ("fiq", 12.0)):
            truth = y[target]
            for arm, noise in (("good", 0.3 * spread), ("bad", 1.5 * spread)):
                pred = truth + rng.normal(0, noise, n_test)
                y_cal = rng.normal(truth.mean(), spread, 60)
                cal = (y_cal, y_cal + rng.normal(0, noise, 60))
                store.add(arm, target, 100, seed, pred, cal)
            store.add("const", target, 100, seed, np.full(n_test, truth.mean() + 2.0))
    return store


def build(tmp_path):
    for fold, n, off in (("A", 60, 1), ("B", 50, 2), ("C", 40, 3)):
        make_store(fold, n, off).save(str(tmp_path))
    return load_mode(str(tmp_path), "loso")


def test_summary_ranks_arms_and_gives_intervals_around_the_mean(tmp_path):
    summary = summarise_mode(build(tmp_path), n_boot=200, pairs=(("good", "bad"),))
    assert summary["n_subjects"] == 150
    good = summary["arms"]["age"]["good"]["100"]["pooled_r2"]
    bad = summary["arms"]["age"]["bad"]["100"]["pooled_r2"]
    assert good["mean"] > bad["mean"]
    assert good["lo"] <= good["mean"] <= good["hi"]
    assert good["sd"] >= 0
    macro = summary["arms"]["age"]["good"]["100"]["macro_r2"]
    assert macro["lo"] < macro["hi"]


def test_paired_difference_is_positive_with_an_interval_above_zero(tmp_path):
    summary = summarise_mode(build(tmp_path), n_boot=200, pairs=(("good", "bad"),))
    diff = summary["diffs"]["age"]["good - bad"]["100"]["pooled_r2"]
    assert diff["mean"] > 0 and diff["lo"] > 0


def test_coverage_and_width_only_exist_for_arms_with_intervals(tmp_path):
    summary = summarise_mode(build(tmp_path), n_boot=100, pairs=())
    good = summary["arms"]["age"]["good"]["100"]
    assert 0.5 < good["cover90_pooled"]["mean"] <= 1.0
    assert good["width90"]["mean"] > good["width80"]["mean"] > 0
    assert "cover90_pooled" not in summary["arms"]["age"]["const"]["100"]


def test_summary_is_strict_json_and_deterministic(tmp_path):
    folds = build(tmp_path)
    a = summarise_mode(folds, n_boot=50, seed=3, pairs=())
    b = summarise_mode(folds, n_boot=50, seed=3, pairs=())
    assert a == b
    json.dumps(a, allow_nan=False)


def test_clean_replaces_non_finite_numbers():
    assert clean({"a": float("nan"), "b": [1.0], "c": {"d": float("inf")}}) == {
        "a": None,
        "b": [1.0],
        "c": {"d": None},
    }


def test_fold_means_average_over_seeds():
    rows = [
        {
            "mode": "loso",
            "fold": "A",
            "arm": "x",
            "target": "age",
            "n": 100,
            "seed": s,
            "r2": float(s),
            "mae": 1.0,
            "spearman": 0.5,
        }
        for s in range(3)
    ]
    out = fold_means(rows)
    assert len(out) == 1
    assert out[0]["r2"] == 1.0 and out[0]["n_seeds"] == 3
    assert out[0]["cover90"] is None
