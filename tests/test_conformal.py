import numpy as np
import pytest

from connpre.conformal import conformal_q, score_run


def test_conformal_q_uses_the_ceil_rank():
    res = np.arange(1, 101, dtype=float)
    assert conformal_q(res, 0.9) == 91.0
    assert conformal_q(res, 0.8) == 81.0
    assert conformal_q(np.arange(1.0, 10.0), 0.95) == float("inf")


def test_coverage_is_close_to_the_level_for_exchangeable_data():
    rng = np.random.default_rng(0)
    covers = []
    for _ in range(300):
        scale = rng.uniform(0.5, 3.0, size=300)
        err = rng.standard_normal(300) * scale
        q = conformal_q(np.abs(err[:100]), 0.9)
        covers.append(np.mean(np.abs(err[100:]) <= q))
    assert np.mean(covers) == pytest.approx(0.9, abs=0.02)


def test_coverage_falls_when_the_test_errors_are_larger():
    rng = np.random.default_rng(1)
    cal = np.abs(rng.standard_normal(100))
    q = conformal_q(cal, 0.9)
    shifted = np.abs(2.0 * rng.standard_normal(2000))
    assert np.mean(shifted <= q) < 0.8


def test_score_run_reports_width_twice_the_radius_and_consistent_coverage():
    rng = np.random.default_rng(2)
    y_cal = rng.normal(0, 1, 100)
    p_cal = y_cal + rng.normal(0, 0.5, 100)
    y_te = rng.normal(0, 1, 50)
    p_te = y_te + rng.normal(0, 0.5, 50)
    row = score_run(y_te, p_te, y_cal, p_cal)
    assert row["width90"] == pytest.approx(2 * row["q90"])
    assert row["width80"] < row["width90"]
    assert row["cover80"] <= row["cover90"]
    assert row["cover90"] == pytest.approx(np.mean(np.abs(y_te - p_te) <= row["q90"]))


def test_score_run_without_calibration_has_only_point_metrics():
    y = np.arange(10, dtype=float)
    row = score_run(y, y + 1.0)
    assert set(row) == {"r2", "mae", "spearman"}
    assert row["mae"] == pytest.approx(1.0)
    assert row["spearman"] == pytest.approx(1.0)
