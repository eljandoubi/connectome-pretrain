"""Cohort container, per-subject normalisation, synthetic cohorts and site summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ARRAYS = ("site", "age", "fiq", "dx", "sex", "sub_id", "n_frames", "n_flat")


@dataclass
class Cohort:
    """Subjects with ROI series (n, T, p), z-scored per subject and ROI, plus labels."""

    x: np.ndarray
    site: np.ndarray
    age: np.ndarray
    fiq: np.ndarray
    dx: np.ndarray
    sex: np.ndarray
    sub_id: np.ndarray
    n_frames: np.ndarray
    n_flat: np.ndarray

    def __len__(self) -> int:
        return len(self.site)

    def target(self, name: str) -> np.ndarray:
        return {"age": self.age, "fiq": self.fiq}[name]

    def subset(self, idx: np.ndarray) -> Cohort:
        return Cohort(self.x[idx], *(getattr(self, k)[idx] for k in _ARRAYS))


def zscore_series(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Z-score every ROI over time inside each subject; a flat ROI becomes zeros."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    flat = sd[:, 0, :] <= eps
    safe = np.where(flat[:, None, :], 1.0, sd)
    z = np.where(flat[:, None, :], 0.0, (x - mu) / safe)
    return z.astype(np.float32), flat.sum(axis=1).astype(np.int64)


def build_cohort(series: list, table: dict, t0: int) -> tuple[Cohort, dict]:
    """Filter subjects, keep the first t0 frames, z-score; returns cohort and counts."""
    n = len(series)
    n_frames = np.array([s.shape[0] for s in series])
    fiq = np.asarray(table["fiq"], dtype=np.float64)
    age = np.asarray(table["age"], dtype=np.float64)
    has_fiq = np.isfinite(fiq) & (fiq > 0)
    has_age = np.isfinite(age) & (age > 0)
    long_enough = n_frames >= t0
    keep = has_fiq & has_age & long_enough
    report = {
        "n_downloaded": int(n),
        "n_no_fiq": int((~has_fiq).sum()),
        "n_no_age_after": int((has_fiq & ~has_age).sum()),
        "n_short_after": int((has_fiq & has_age & ~long_enough).sum()),
        "n_kept": int(keep.sum()),
        "t0": int(t0),
    }
    pos = np.flatnonzero(keep)
    raw = np.stack([np.asarray(series[i][:t0], dtype=np.float64) for i in pos])
    z, n_flat = zscore_series(raw)
    report["n_with_flat_roi"] = int((n_flat > 0).sum())
    cohort = Cohort(
        x=z,
        site=np.asarray(table["site"])[pos].astype(str),
        age=age[pos],
        fiq=fiq[pos],
        dx=np.asarray(table["dx"])[pos].astype(np.int64),
        sex=np.asarray(table["sex"])[pos].astype(np.int64),
        sub_id=np.asarray(table["sub_id"])[pos].astype(np.int64),
        n_frames=n_frames[pos],
        n_flat=n_flat,
    )
    return cohort, report


def save_cohort(path: str, cohort: Cohort) -> None:
    """Write one compressed npz; the z-scored series are stored in float16."""
    arrays = {k: getattr(cohort, k) for k in _ARRAYS}
    np.savez_compressed(path, x=cohort.x.astype(np.float16), **arrays)


def load_cohort(path: str) -> Cohort:
    with np.load(path, allow_pickle=False) as f:
        return Cohort(f["x"].astype(np.float32), *(f[k] for k in _ARRAYS))


def lag1_autocorr(x: np.ndarray) -> np.ndarray:
    """Mean lag-1 autocorrelation over ROIs, per subject, of z-scored series."""
    return (x[:, :-1, :] * x[:, 1:, :]).mean(axis=1).mean(axis=1)


def site_table(cohort: Cohort) -> list[dict]:
    """One row per site: size, label summaries, original length, lag-1 autocorr."""
    lag = lag1_autocorr(cohort.x)
    rows = []
    for s in np.unique(cohort.site):
        m = cohort.site == s
        rows.append(
            {
                "site": str(s),
                "n": int(m.sum()),
                "age_mean": float(cohort.age[m].mean()),
                "age_sd": float(cohort.age[m].std()),
                "fiq_mean": float(cohort.fiq[m].mean()),
                "fiq_sd": float(cohort.fiq[m].std()),
                "share_dx1": float((cohort.dx[m] == 1).mean()),
                "share_sex1": float((cohort.sex[m] == 1).mean()),
                "frames_min": int(cohort.n_frames[m].min()),
                "frames_median": float(np.median(cohort.n_frames[m])),
                "lag1": float(lag[m].mean()),
            }
        )
    return sorted(rows, key=lambda r: (-r["n"], r["site"]))


def _ar1(rng: np.random.Generator, shape: tuple, rho: float) -> np.ndarray:
    out = np.empty(shape)
    out[..., 0, :] = rng.standard_normal(shape[:-2] + shape[-1:])
    for t in range(1, shape[-2]):
        noise = rng.standard_normal(shape[:-2] + shape[-1:])
        out[..., t, :] = rho * out[..., t - 1, :] + np.sqrt(1 - rho**2) * noise
    return out


def synthetic_cohort(
    sizes: tuple[int, ...] = (40, 40, 30, 30, 20),
    n_roi: int = 24,
    t: int = 96,
    seed: int = 0,
    n_factors: int = 4,
) -> Cohort:
    """Latent-factor series whose loadings move with age (strong) and a trait (weak).

    Each site shifts the mean age, the mean score, the noise level and the loadings,
    so a leave-one-site-out split meets a real shift.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_roi, n_factors)) / np.sqrt(n_factors)
    d_age = rng.standard_normal((n_roi, n_factors)) / np.sqrt(n_factors)
    d_trait = rng.standard_normal((n_roi, n_factors)) / np.sqrt(n_factors)
    series, rows = [], []
    for k, size in enumerate(sizes):
        d_site = 0.3 * rng.standard_normal((n_roi, n_factors)) / np.sqrt(n_factors)
        age_mu, noise_sd = 10 + 4 * k, 0.8 + 0.15 * k
        for _ in range(size):
            age = float(np.clip(rng.normal(age_mu, 4.0), 6.0, 60.0))
            trait = rng.standard_normal()
            load = base + d_site + (age - 15) / 8 * d_age + 0.5 * trait * d_trait
            f = _ar1(rng, (t, n_factors), 0.5)
            noise = _ar1(rng, (t, n_roi), 0.3)
            series.append(f @ load.T + noise_sd * noise)
            fiq = 100 + 3.0 * (k - 2) + 8.0 * trait + rng.normal(0, 10)
            rows.append(
                (f"S{k}", age, fiq, 1 + (trait > 0.5), 1 + int(rng.random() < 0.2))
            )
    table = {
        "site": [r[0] for r in rows],
        "age": [r[1] for r in rows],
        "fiq": [r[2] for r in rows],
        "dx": [r[3] for r in rows],
        "sex": [r[4] for r in rows],
        "sub_id": list(range(len(rows))),
    }
    return build_cohort(series, table, t)[0]
