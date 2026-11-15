import numpy as np
import pytest

from connpre.data import synthetic_cohort
from connpre.features import (
    TangentSpace,
    correlation_vectors,
    geometric_mean,
    ledoit_wolf_covs,
    log_euclidean_mean,
    sym_to_vec,
)


def random_spd(n, p, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, p, p + 3))
    return a @ np.swapaxes(a, 1, 2) / (p + 3) + 0.1 * np.eye(p)


def test_correlation_vectors_match_numpy():
    cohort = synthetic_cohort(sizes=(4, 3), n_roi=10, t=40)
    vec = correlation_vectors(cohort.x)
    iu = np.triu_indices(10, 1)
    for i in range(len(cohort)):
        expected = np.corrcoef(cohort.x[i], rowvar=False)[iu]
        assert np.allclose(vec[i], expected, atol=1e-5)
    assert vec.shape == (7, 45)


def test_correlation_vectors_are_zero_for_a_flat_roi():
    x = np.random.default_rng(0).standard_normal((2, 30, 5)).astype(np.float32)
    x[:, :, 1] = 0.0
    x = x - x.mean(axis=1, keepdims=True)
    x[:, :, [0, 2, 3, 4]] /= x[:, :, [0, 2, 3, 4]].std(axis=1, keepdims=True)
    vec = correlation_vectors(x)
    iu = np.triu_indices(5, 1)
    touching = (iu[0] == 1) | (iu[1] == 1)
    assert np.all(vec[:, touching] == 0)


def test_ledoit_wolf_covariances_are_positive_definite_when_t_is_below_p():
    cohort = synthetic_cohort(sizes=(3, 3), n_roi=30, t=20)
    covs = ledoit_wolf_covs(cohort.x)
    assert covs.shape == (6, 30, 30)
    assert np.allclose(covs, np.swapaxes(covs, 1, 2))
    assert np.linalg.eigvalsh(covs).min() > 1e-6


def test_geometric_mean_of_diagonal_matrices_is_the_log_mean():
    rng = np.random.default_rng(1)
    d = rng.uniform(0.5, 3.0, size=(12, 6))
    covs = np.stack([np.diag(v) for v in d])
    g, _ = geometric_mean(covs)
    assert np.allclose(np.diag(g), np.exp(np.log(d).mean(axis=0)), atol=1e-6)
    assert np.allclose(g - np.diag(np.diag(g)), 0, atol=1e-8)


def test_log_euclidean_start_is_exact_for_commuting_matrices():
    rng = np.random.default_rng(4)
    d = rng.uniform(0.5, 3.0, size=(9, 5))
    covs = np.stack([np.diag(v) for v in d])
    g, n_iter = geometric_mean(covs, tol=1e-9)
    assert np.allclose(g, log_euclidean_mean(covs), atol=1e-10)
    assert n_iter == 0


def test_a_looser_tolerance_stops_earlier_and_stays_close():
    covs = random_spd(15, 8, seed=7)
    tight, n_tight = geometric_mean(covs, tol=1e-9)
    loose, n_loose = geometric_mean(covs, tol=1e-3)
    assert n_loose <= n_tight
    assert np.linalg.norm(tight - loose) < 5e-3


def test_tangent_vectors_of_the_fit_set_average_to_zero():
    covs = random_spd(20, 8)
    ts = TangentSpace().fit(covs)
    mean_tangent = ts.transform_matrices(covs).mean(axis=0)
    assert np.linalg.norm(mean_tangent) < 1e-5
    assert np.abs(ts.transform_matrices(ts.ref_[None])).max() < 1e-8


def test_tangent_of_diagonal_matrices_has_the_closed_form():
    rng = np.random.default_rng(2)
    d = rng.uniform(0.5, 3.0, size=(10, 5))
    covs = np.stack([np.diag(v) for v in d])
    ts = TangentSpace().fit(covs)
    mats = ts.transform_matrices(covs)
    ref = np.exp(np.log(d).mean(axis=0))
    for i in range(10):
        assert np.allclose(np.diag(mats[i]), np.log(d[i] / ref), atol=1e-6)


def test_vectorisation_keeps_the_frobenius_norm():
    covs = random_spd(5, 7)
    mats = TangentSpace().fit(covs).transform_matrices(covs)
    vec = sym_to_vec(mats)
    assert vec.shape == (5, 28)
    assert np.allclose(np.linalg.norm(vec, axis=1), np.linalg.norm(mats, axis=(1, 2)))


def test_transform_gives_vectors_of_the_right_size_and_is_fit_set_independent():
    covs = random_spd(12, 6, seed=3)
    ts = TangentSpace().fit(covs[:8])
    assert ts.transform(covs[8:]).shape == (4, 21)
    assert np.array_equal(ts.transform(covs[8:]), ts.transform(covs)[8:])


def test_tangent_agrees_with_nilearn_when_installed():
    connectome = pytest.importorskip("nilearn.connectome")
    from sklearn.covariance import LedoitWolf

    cohort = synthetic_cohort(sizes=(8, 8), n_roi=12, t=60)
    covs = ledoit_wolf_covs(cohort.x)
    mine = TangentSpace(tol=1e-9).fit(covs).transform_matrices(covs)
    measure = connectome.ConnectivityMeasure(
        kind="tangent", cov_estimator=LedoitWolf(store_precision=False), vectorize=False
    )
    theirs = measure.fit_transform(list(cohort.x.astype(np.float64)))
    assert np.abs(mine - theirs).max() < 1e-4
