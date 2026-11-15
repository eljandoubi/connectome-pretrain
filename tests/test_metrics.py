import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from connpre.metrics import (
    mae,
    mean_bank,
    metric_bank,
    percentile_ci,
    r2,
    site_parts,
    spearman,
)


def toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(20, 5, n)
    p = y + rng.normal(0, 3, n)
    return y, p


def test_point_metrics_match_reference_implementations():
    y, p = toy()
    assert r2(y, p) == pytest.approx(r2_score(y, p))
    assert mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert spearman(y, p) == pytest.approx(spearmanr(y, p).statistic)


def test_spearman_is_zero_for_a_constant_prediction():
    y, _ = toy()
    assert spearman(y, np.full_like(y, 3.0)) == 0.0


def test_metrics_vectorise_over_leading_axes():
    y, p = toy()
    stack = np.stack([p, p + 1, p * 0 + y.mean()])
    out = r2(y, stack)
    assert out.shape == (3,)
    assert out[2] == pytest.approx(0.0, abs=1e-12)
    assert out[0] == pytest.approx(r2_score(y, p))


def test_metric_bank_macro_is_the_mean_of_site_scores():
    y, p = toy(90)
    site = np.array(["a"] * 40 + ["b"] * 30 + ["c"] * 20)
    parts = site_parts(site)
    bank = metric_bank(y, p, parts)
    per = [r2_score(y[site == s], p[site == s]) for s in "abc"]
    assert bank["macro_r2"][0] == pytest.approx(np.mean(per))
    assert bank["pooled_r2"][0] == pytest.approx(r2_score(y, p))
    assert bank["macro_mae"][0] == pytest.approx(
        np.mean([mean_absolute_error(y[site == s], p[site == s]) for s in "abc"])
    )


def test_bootstrap_parts_stay_inside_their_site():
    site = np.array(["a"] * 10 + ["b"] * 6)
    parts = site_parts(site, n_boot=20, rng=np.random.default_rng(0))
    assert [x.shape for x in parts] == [(20, 10), (20, 6)]
    assert parts[0].max() < 10 and parts[1].min() >= 10


def test_bootstrap_interval_covers_the_point_estimate_and_is_paired():
    y, p = toy(120, seed=1)
    site = np.array(["a"] * 70 + ["b"] * 50)
    point = metric_bank(y, np.stack([p, p]), site_parts(site))["macro_r2"].mean()
    parts = site_parts(site, n_boot=300, rng=np.random.default_rng(2))
    both = metric_bank(y, np.stack([p, p]), parts)["macro_r2"]
    lo, hi = percentile_ci(both.mean(axis=0))
    assert lo < point < hi
    assert np.array_equal(both[0], both[1])


def test_mean_bank_gives_macro_and_pooled_coverage():
    inside = np.array([1, 1, 1, 1, 0, 0], dtype=float)
    site = np.array(["a"] * 4 + ["b"] * 2)
    bank = mean_bank(inside, site_parts(site))
    assert bank["macro"][0] == pytest.approx(0.5)
    assert bank["pooled"][0] == pytest.approx(4 / 6)
