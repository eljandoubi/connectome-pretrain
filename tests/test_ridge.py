import numpy as np
from sklearn.linear_model import Ridge

from connpre.ridge import (
    REL_ALPHAS,
    center_kernel,
    group_folds,
    rbf_cv_predict,
    ridge_cv_predict,
    ridge_path,
)


def linear_problem(n=100, p=30, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    w = rng.standard_normal(p)
    return x, x @ w + noise * rng.standard_normal(n)


def test_centered_training_kernel_has_zero_row_sums():
    x, _ = linear_problem(n=20)
    k = x @ x.T
    kc, _ = center_kernel(k, k[:5])
    assert np.allclose(kc.sum(axis=1), 0, atol=1e-8)


def test_ridge_path_matches_sklearn_ridge_with_intercept():
    x, y = linear_problem(n=60, p=15)
    y = y + 5.0
    tr, te = np.arange(40), np.arange(40, 60)
    k = x @ x.T
    path = ridge_path(k[np.ix_(tr, tr)], y[tr, None], k[np.ix_(te, tr)])
    xc = x[tr] - x[tr].mean(axis=0)
    scale = float(np.mean(np.sum(xc**2, axis=1)))
    for i in (0, 6, 12):
        model = Ridge(alpha=REL_ALPHAS[i] * scale, fit_intercept=True).fit(x[tr], y[tr])
        assert np.allclose(path[i, :, 0], model.predict(x[te]), atol=1e-6)


def test_ridge_path_columns_equal_separate_single_target_fits():
    x, y = linear_problem(n=50, p=10)
    y2 = np.stack([y, -2 * y + 1], axis=1)
    k = x @ x.T
    tr, te = np.arange(35), np.arange(35, 50)
    both = ridge_path(k[np.ix_(tr, tr)], y2[tr], k[np.ix_(te, tr)])
    one = ridge_path(k[np.ix_(tr, tr)], y2[tr][:, [1]], k[np.ix_(te, tr)])
    assert np.allclose(both[:, :, 1], one[:, :, 0])


def test_group_folds_never_split_a_group():
    groups = np.repeat(np.arange(6), [10, 8, 7, 6, 5, 4])
    folds = list(group_folds(groups, n_splits=3))
    assert len(folds) == 3
    for tr, va in folds:
        assert set(groups[tr]).isdisjoint(set(groups[va]))
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(40))


def test_group_folds_fall_back_to_plain_kfold_with_one_group():
    folds = list(group_folds(np.zeros(20, dtype=int), n_splits=4))
    assert len(folds) == 4


def test_cv_picks_weak_regularisation_for_signal_and_strong_for_noise():
    x, y = linear_problem(n=120, p=20, noise=0.05)
    k = x @ x.T
    lab = np.arange(100)
    groups = np.arange(100) % 5
    _, best_signal = ridge_cv_predict(k, y[:, None], lab, np.arange(100, 120), groups)
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((120, 1))
    _, best_noise = ridge_cv_predict(k, noise, lab, np.arange(100, 120), groups)
    assert best_signal[0] < len(REL_ALPHAS) // 2
    assert best_noise[0] >= len(REL_ALPHAS) // 2


def test_ridge_cv_predict_generalises_to_held_out_subjects():
    x, y = linear_problem(n=140, p=25, noise=0.3, seed=2)
    k = x @ x.T
    lab, te = np.arange(100), np.arange(100, 140)
    pred, _ = ridge_cv_predict(k, y[:, None], lab, te, np.arange(100) % 5)
    r2 = 1 - ((y[te] - pred[:, 0]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    assert r2 > 0.9


def test_rbf_kernel_ridge_fits_a_nonlinear_target():
    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 3, size=(160, 2))
    y = x[:, 0] ** 2 / 3 + 0.05 * rng.standard_normal(160)
    gram = x @ x.T
    lab, te = np.arange(120), np.arange(120, 160)
    groups = np.arange(120) % 5
    pred, chosen = rbf_cv_predict(gram, y[:, None], lab, te, groups, ref=np.arange(160))
    lin, _ = ridge_cv_predict(gram, y[:, None], lab, te, groups)

    def r2(p):
        return 1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()

    assert r2(pred[:, 0]) > 0.7
    assert r2(lin[:, 0]) < 0.3
    assert chosen.shape == (1, 2)
