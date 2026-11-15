"""Metrics, per-site aggregation and bootstrap resampling over subjects."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def r2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """R2 over the last axis; y broadcasts against p."""
    ss_res = ((y - p) ** 2).sum(axis=-1)
    ss_tot = ((y - y.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    return 1.0 - ss_res / ss_tot


def mae(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.abs(y - p).mean(axis=-1)


def spearman(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Spearman correlation over the last axis; 0 when either side is constant."""
    ry = rankdata(y, axis=-1)
    rp = rankdata(p, axis=-1)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    rp = rp - rp.mean(axis=-1, keepdims=True)
    den = np.sqrt((ry**2).sum(axis=-1) * (rp**2).sum(axis=-1))
    num = (ry * rp).sum(axis=-1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def site_parts(
    site: np.ndarray, n_boot: int = 0, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    """Index arrays (replicates, n_site) per site: identity, or resampled in site."""
    parts = []
    for s in np.unique(site):
        pos = np.flatnonzero(site == s)
        if n_boot == 0:
            parts.append(pos[None, :])
        else:
            parts.append(pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))])
    return parts


def metric_bank(y: np.ndarray, p: np.ndarray, parts: list[np.ndarray]) -> dict:
    """Macro (mean over sites) and pooled metrics, one value per replicate.

    y is (n,), p is (..., n); each part is (replicates, n_site). Outputs are
    (..., replicates).
    """
    per = {"r2": [], "mae": [], "spearman": []}
    for idx in parts:
        yy, pp = y[idx], p[..., idx]
        per["r2"].append(r2(yy, pp))
        per["mae"].append(mae(yy, pp))
        per["spearman"].append(spearman(yy, pp))
    every = np.concatenate(parts, axis=-1)
    yy, pp = y[every], p[..., every]
    return {
        "macro_r2": np.mean(per["r2"], axis=0),
        "macro_mae": np.mean(per["mae"], axis=0),
        "macro_spearman": np.mean(per["spearman"], axis=0),
        "pooled_r2": r2(yy, pp),
        "pooled_mae": mae(yy, pp),
    }


def mean_bank(values: np.ndarray, parts: list[np.ndarray]) -> dict:
    """Macro and pooled means of a per-subject quantity (for example coverage)."""
    per = [values[..., idx].mean(axis=-1) for idx in parts]
    every = np.concatenate(parts, axis=-1)
    return {"macro": np.mean(per, axis=0), "pooled": values[..., every].mean(axis=-1)}


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
