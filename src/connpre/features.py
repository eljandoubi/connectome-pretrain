"""Connectivity features: correlation vectors, Ledoit-Wolf, tangent space."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import ledoit_wolf


def correlation_vectors(x: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Upper triangle without the diagonal of each subject's correlation matrix.

    The series are z-scored, so the correlation is x^T x / T; a flat ROI gives zeros.
    """
    n, t, p = x.shape
    iu = np.triu_indices(p, 1)
    out = np.empty((n, len(iu[0])), dtype=np.float32)
    for s in range(0, n, chunk):
        xs = x[s : s + chunk].astype(np.float64)
        corr = np.einsum("ntp,ntq->npq", xs, xs) / t
        out[s : s + chunk] = corr[:, iu[0], iu[1]]
    return out


def ledoit_wolf_covs(x: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of every subject, shape (n, p, p), float64."""
    n, _, p = x.shape
    out = np.empty((n, p, p))
    for i in range(n):
        out[i] = ledoit_wolf(x[i].astype(np.float64), assume_centered=True)[0]
    return out


def _sym_apply(vals: np.ndarray, vecs: np.ndarray, fn) -> np.ndarray:
    return (vecs * fn(vals)[..., None, :]) @ np.swapaxes(vecs, -1, -2)


def _log_whitened(covs: np.ndarray, w: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(w @ covs @ w)
    return _sym_apply(vals, vecs, np.log)


def log_euclidean_mean(covs: np.ndarray, chunk: int = 64) -> np.ndarray:
    """exp of the mean matrix logarithm; a close start for the Karcher mean."""
    acc = np.zeros(covs.shape[1:])
    for s in range(0, len(covs), chunk):
        vals, vecs = np.linalg.eigh(covs[s : s + chunk])
        acc += _sym_apply(vals, vecs, np.log).sum(axis=0)
    vals, vecs = np.linalg.eigh(acc / len(covs))
    return _sym_apply(vals, vecs, np.exp)


def geometric_mean(
    covs: np.ndarray, max_iter: int = 50, tol: float = 1e-6, chunk: int = 64
) -> tuple[np.ndarray, int]:
    """Riemannian (Karcher) mean of SPD matrices by fixed-point iteration.

    Stops when the Frobenius norm of the mean tangent vector is below tol.
    """
    g = log_euclidean_mean(covs, chunk)
    for it in range(max_iter):
        vals, vecs = np.linalg.eigh(g)
        g_isqrt = _sym_apply(vals, vecs, lambda v: v**-0.5)
        g_sqrt = _sym_apply(vals, vecs, np.sqrt)
        mean_log = np.zeros_like(g)
        for s in range(0, len(covs), chunk):
            mean_log += _log_whitened(covs[s : s + chunk], g_isqrt).sum(axis=0)
        mean_log /= len(covs)
        if np.linalg.norm(mean_log) < tol:
            return g, it
        lv, lw = np.linalg.eigh((mean_log + mean_log.T) / 2)
        g = g_sqrt @ _sym_apply(lv, lw, np.exp) @ g_sqrt
        g = (g + g.T) / 2
    return g, max_iter


def sym_to_vec(mats: np.ndarray) -> np.ndarray:
    """Upper triangle with the diagonal; off-diagonal entries times sqrt(2)."""
    p = mats.shape[-1]
    iu = np.triu_indices(p)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return mats[..., iu[0], iu[1]] * scale


class TangentSpace:
    """Project SPD matrices to the tangent space at the geometric mean of a fit set."""

    def __init__(self, max_iter: int = 50, tol: float = 1e-6, chunk: int = 64):
        self.max_iter, self.tol, self.chunk = max_iter, tol, chunk

    def fit(self, covs: np.ndarray) -> TangentSpace:
        self.ref_, self.n_iter_ = geometric_mean(
            covs, self.max_iter, self.tol, self.chunk
        )
        vals, vecs = np.linalg.eigh(self.ref_)
        self.whiten_ = _sym_apply(vals, vecs, lambda v: v**-0.5)
        return self

    def transform_matrices(self, covs: np.ndarray) -> np.ndarray:
        out = np.empty_like(covs)
        for s in range(0, len(covs), self.chunk):
            out[s : s + self.chunk] = _log_whitened(
                covs[s : s + self.chunk], self.whiten_
            )
        return out

    def transform(self, covs: np.ndarray) -> np.ndarray:
        p = covs.shape[-1]
        out = np.empty((len(covs), p * (p + 1) // 2), dtype=np.float32)
        for s in range(0, len(covs), self.chunk):
            mats = _log_whitened(covs[s : s + self.chunk], self.whiten_)
            out[s : s + self.chunk] = sym_to_vec(mats)
        return out
