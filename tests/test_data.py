import numpy as np

from connpre.data import (
    Cohort,
    build_cohort,
    load_cohort,
    save_cohort,
    site_table,
    synthetic_cohort,
    zscore_series,
)


def test_zscore_has_zero_mean_unit_variance_and_zeroes_flat_rois():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 50, 6)) * 4 + 2
    x[1, :, 2] = 7.0
    z, n_flat = zscore_series(x)
    assert z.shape == x.shape
    ok = np.ones((3, 6), dtype=bool)
    ok[1, 2] = False
    assert np.allclose(z.mean(axis=1)[ok], 0, atol=1e-6)
    assert np.allclose(z.std(axis=1)[ok], 1, atol=1e-5)
    assert np.all(z[1, :, 2] == 0)
    assert n_flat.tolist() == [0, 1, 0]


def test_build_cohort_filters_in_order_and_truncates():
    rng = np.random.default_rng(1)
    lengths = [60, 40, 60, 60, 60]
    series = [rng.standard_normal((n, 5)) for n in lengths]
    table = {
        "site": ["A", "A", "B", "B", "A"],
        "age": [10.0, 11.0, 12.0, np.nan, 14.0],
        "fiq": [100.0, 90.0, -9999.0, 80.0, 70.0],
        "dx": [1, 2, 1, 2, 1],
        "sex": [1, 1, 2, 2, 1],
        "sub_id": [10, 11, 12, 13, 14],
    }
    cohort, report = build_cohort(series, table, t0=50)
    assert cohort.x.shape == (2, 50, 5)
    assert cohort.sub_id.tolist() == [10, 14]
    assert report["n_downloaded"] == 5
    assert report["n_no_fiq"] == 1
    assert report["n_no_age_after"] == 1
    assert report["n_short_after"] == 1
    assert report["n_kept"] == 2
    assert cohort.n_frames.tolist() == [60, 60]


def test_save_and_load_round_trip(tmp_path):
    cohort = synthetic_cohort(sizes=(6, 5), n_roi=8, t=32)
    path = str(tmp_path / "c.npz")
    save_cohort(path, cohort)
    back = load_cohort(path)
    assert back.x.shape == cohort.x.shape
    assert np.allclose(back.x, cohort.x, atol=2e-2)
    assert back.site.tolist() == cohort.site.tolist()
    assert np.array_equal(back.age, cohort.age)


def test_subset_and_target_follow_the_index():
    cohort = synthetic_cohort(sizes=(6, 5), n_roi=8, t=32)
    sub = cohort.subset(np.array([0, 3, 7]))
    assert isinstance(sub, Cohort)
    assert len(sub) == 3
    assert np.array_equal(sub.target("age"), cohort.age[[0, 3, 7]])
    assert np.array_equal(sub.x, cohort.x[[0, 3, 7]])


def test_synthetic_cohort_shifts_age_by_site():
    cohort = synthetic_cohort(sizes=(40, 40, 40), n_roi=10, t=40, seed=3)
    means = [cohort.age[cohort.site == s].mean() for s in ("S0", "S1", "S2")]
    assert means[0] < means[1] < means[2]
    assert np.isfinite(cohort.x).all()


def test_site_table_is_sorted_by_size():
    cohort = synthetic_cohort(sizes=(5, 12, 8), n_roi=8, t=32)
    rows = site_table(cohort)
    assert [r["n"] for r in rows] == [12, 8, 5]
    assert rows[0]["site"] == "S1"
    assert all(-1 <= r["lag1"] <= 1 for r in rows)
