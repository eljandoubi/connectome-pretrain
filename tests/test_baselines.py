from dataclasses import replace

import numpy as np
import pytest

from connpre.baselines import fold_baselines, site_mean_prediction
from connpre.data import synthetic_cohort
from connpre.features import correlation_vectors, ledoit_wolf_covs
from connpre.protocol import N_ALL, insite_folds, loso_folds


@pytest.fixture(scope="module")
def setup():
    cohort = synthetic_cohort(sizes=(50, 40, 40, 30, 20), n_roi=20, t=90, seed=4)
    return cohort, ledoit_wolf_covs(cohort.x), correlation_vectors(cohort.x)


def run(setup, mode="loso", fold_index=1, sizes=(30, N_ALL)):
    cohort, covs, corr = setup
    if mode == "loso":
        fold = loso_folds(cohort.site, 30)[fold_index]
    else:
        fold = insite_folds(cohort.site, 5, seed=0)[fold_index]
    return fold, fold_baselines(cohort, fold, covs, corr, mode, (0, 1), sizes, 20)


def mean_metric(store, arm, target, n, key):
    rows = [
        r for r in store.rows if (r["arm"], r["target"], r["n"]) == (arm, target, n)
    ]
    return float(np.mean([r[key] for r in rows]))


def test_loso_run_has_every_arm_target_size_and_seed(setup):
    _, store = run(setup)
    arms = {r["arm"] for r in store.rows}
    assert arms == {
        "corr_ridge",
        "tangent_ridge",
        "tangent_krr",
        "site_const",
        "site_oracle",
    }
    assert len(store.rows) == 5 * 2 * 2 * 2
    assert all(r["n_test"] == len(store.test_idx) for r in store.rows)


def test_tangent_ridge_predicts_age_better_than_the_constant(setup):
    _, store = run(setup)
    ridge = mean_metric(store, "tangent_ridge", "age", N_ALL, "r2")
    const = mean_metric(store, "site_const", "age", N_ALL, "r2")
    assert ridge > const + 0.2
    assert mean_metric(store, "tangent_ridge", "age", N_ALL, "spearman") > 0.5


def test_oracle_row_has_zero_within_site_r2_and_no_conformal_fields(setup):
    _, store = run(setup)
    row = next(r for r in store.rows if r["arm"] == "site_oracle")
    assert row["r2"] == pytest.approx(0.0, abs=1e-9)
    assert "q90" not in row


def test_insite_mode_swaps_the_oracle_for_the_site_mean(setup):
    _, store = run(setup, mode="insite", fold_index=0)
    arms = {r["arm"] for r in store.rows}
    assert "site_mean" in arms and "site_oracle" not in arms


def test_the_held_out_series_do_not_change_calibration_radii(setup):
    cohort, covs, corr = setup
    fold = loso_folds(cohort.site, 30)[1]
    base = fold_baselines(cohort, fold, covs, corr, "loso", (0,), (30,), 20)
    x = cohort.x.copy()
    x[fold.test] *= 5.0
    twisted = replace(cohort, x=x)
    covs2 = ledoit_wolf_covs(twisted.x)
    corr2 = correlation_vectors(twisted.x)
    other = fold_baselines(twisted, fold, covs2, corr2, "loso", (0,), (30,), 20)
    for a, b in zip(base.rows, other.rows, strict=True):
        if a["arm"] in ("tangent_ridge", "tangent_krr", "corr_ridge"):
            assert a["q90"] == pytest.approx(b["q90"], rel=1e-6)


def test_site_mean_prediction_uses_the_own_site_and_falls_back_to_the_mean():
    lab_site = np.array(["a", "a", "b"])
    y = np.array([1.0, 3.0, 10.0])
    out = site_mean_prediction(lab_site, y, np.array(["a", "b", "c"]))
    assert out.tolist() == [2.0, 10.0, pytest.approx(14 / 3)]
