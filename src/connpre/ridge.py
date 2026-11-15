"""Kernel ridge in dual form with a free intercept and alpha chosen by group CV."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

REL_ALPHAS = 10.0 ** np.arange(-3.0, 3.01, 0.5)
GAMMA_MULTS = (0.5, 1.0, 2.0)


def center_kernel(k_tr: np.ndarray, k_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre train and test kernels in feature space with the training mean."""
    col = k_tr.mean(axis=0)
    tot = col.mean()
    kc_tr = k_tr - col[None, :] - col[:, None] + tot
    kc_te = k_te - k_te.mean(axis=1, keepdims=True) - col[None, :] + tot
    return kc_tr, kc_te


def ridge_path(
    k_tr: np.ndarray,
    y_tr: np.ndarray,
    k_te: np.ndarray,
    rel_alphas: np.ndarray = REL_ALPHAS,
) -> np.ndarray:
    """Predictions for every alpha at once, shape (alphas, m, targets).

    Alpha is relative to the mean diagonal of the centred training kernel.
    """
    kc_tr, kc_te = center_kernel(k_tr, k_te)
    w, v = np.linalg.eigh(kc_tr)
    w = np.clip(w, 0.0, None)
    scale = max(float(np.mean(np.diag(kc_tr))), 1e-12)
    ybar = y_tr.mean(axis=0)
    proj = v.T @ (y_tr - ybar)
    ktv = kc_te @ v
    out = np.empty((len(rel_alphas), kc_te.shape[0], y_tr.shape[1]))
    for i, a in enumerate(rel_alphas):
        out[i] = ktv @ (proj / (w + a * scale)[:, None]) + ybar
    return out


def group_folds(groups: np.ndarray, n_splits: int = 5):
    """Group K-fold over sites; plain K-fold when only one group is present."""
    n_groups = len(np.unique(groups))
    dummy = np.zeros(len(groups))
    if n_groups >= 2:
        yield from GroupKFold(n_splits=min(n_splits, n_groups)).split(
            dummy, groups=groups
        )
    else:
        yield from KFold(
            min(n_splits, len(groups)), shuffle=True, random_state=0
        ).split(dummy)


def cv_sse(
    k: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    rel_alphas: np.ndarray = REL_ALPHAS,
    n_splits: int = 5,
) -> np.ndarray:
    """Held-out squared error summed over folds, shape (alphas, targets)."""
    sse = np.zeros((len(rel_alphas), y.shape[1]))
    for tr, va in group_folds(groups, n_splits):
        p = ridge_path(k[np.ix_(tr, tr)], y[tr], k[np.ix_(va, tr)], rel_alphas)
        sse += ((p - y[va][None]) ** 2).sum(axis=1)
    return sse


def ridge_cv_predict(
    kernel: np.ndarray,
    y: np.ndarray,
    lab: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    rel_alphas: np.ndarray = REL_ALPHAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick alpha per target by group CV on lab, refit on lab, predict pred.

    kernel is (n_all, n_all), y is (n_all, targets), groups are the sites of lab.
    Returns predictions (len(pred), targets) and the chosen alpha index per target.
    """
    k_lab = kernel[np.ix_(lab, lab)]
    best = cv_sse(k_lab, y[lab], groups, rel_alphas).argmin(axis=0)
    path = ridge_path(k_lab, y[lab], kernel[np.ix_(pred, lab)], rel_alphas)
    out = np.stack([path[best[t], :, t] for t in range(y.shape[1])], axis=1)
    return out, best


def sq_dists(gram: np.ndarray) -> np.ndarray:
    d = np.diag(gram)
    return np.maximum(d[:, None] + d[None, :] - 2.0 * gram, 0.0)


def rbf_cv_predict(
    gram: np.ndarray,
    y: np.ndarray,
    lab: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    ref: np.ndarray,
    gamma_mults: tuple = GAMMA_MULTS,
    rel_alphas: np.ndarray = REL_ALPHAS,
) -> tuple[np.ndarray, np.ndarray]:
    """RBF kernel ridge; gamma = mult / median squared distance among ref subjects.

    Gamma multiplier and alpha are chosen together per target by group CV.
    Returns predictions and the chosen (gamma index, alpha index) per target.
    """
    d2 = sq_dists(gram)
    sub = d2[np.ix_(ref, ref)]
    med = float(np.median(sub[np.triu_indices(len(ref), 1)]))
    kernels = [np.exp(-(m / med) * d2) for m in gamma_mults]
    sse = np.stack(
        [cv_sse(k[np.ix_(lab, lab)], y[lab], groups, rel_alphas) for k in kernels]
    )
    out = np.empty((len(pred), y.shape[1]))
    chosen = np.empty((y.shape[1], 2), dtype=np.int64)
    for t in range(y.shape[1]):
        gi, ai = np.unravel_index(sse[:, :, t].argmin(), sse.shape[:2])
        k = kernels[gi]
        path = ridge_path(
            k[np.ix_(lab, lab)], y[lab][:, [t]], k[np.ix_(pred, lab)], rel_alphas
        )
        out[:, t] = path[ai, :, 0]
        chosen[t] = (gi, ai)
    return out, chosen
