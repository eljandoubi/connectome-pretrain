"""Split conformal intervals and the per-run score row."""

from __future__ import annotations

import numpy as np

from connpre.metrics import mae, r2, spearman

LEVELS = (0.8, 0.9)


def conformal_q(residuals: np.ndarray, level: float) -> float:
    """Split-conformal radius: the ceil((n + 1) * level)-th smallest residual."""
    n = len(residuals)
    k = int(np.ceil((n + 1) * level))
    return float("inf") if k > n else float(np.sort(residuals)[k - 1])


def score_run(
    y_te: np.ndarray,
    p_te: np.ndarray,
    y_cal: np.ndarray | None = None,
    p_cal: np.ndarray | None = None,
    levels: tuple = LEVELS,
) -> dict:
    """Point metrics on the held-out subjects plus conformal coverage and width.

    Without calibration subjects only the point metrics are returned.
    """
    row = {
        "r2": float(r2(y_te, p_te)),
        "mae": float(mae(y_te, p_te)),
        "spearman": float(spearman(y_te, p_te)),
    }
    if y_cal is None:
        return row
    res = np.abs(y_cal - p_cal)
    for level in levels:
        tag = int(round(level * 100))
        q = conformal_q(res, level)
        row[f"q{tag}"] = q
        row[f"cover{tag}"] = float((np.abs(y_te - p_te) <= q).mean())
        row[f"width{tag}"] = 2.0 * q
    return row
