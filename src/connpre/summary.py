"""Aggregate stored runs into seed means, bootstrap intervals and paired differences."""

from __future__ import annotations

import math

import numpy as np

from connpre.metrics import mean_bank, metric_bank, percentile_ci, site_parts
from connpre.results import keys_of, pooled

METRICS = ("macro_r2", "macro_mae", "macro_spearman", "pooled_r2", "pooled_mae")
LEVEL_COLUMN = {"80": 0, "90": 1}
PAIRS = (
    ("ft_pre", "ft_scratch"),
    ("ft_pre", "tangent_ridge"),
    ("ft_scratch", "tangent_ridge"),
    ("probe_flat", "tangent_ridge"),
    ("probe_mean", "tangent_ridge"),
    ("probe_flat", "probe_flat_rand"),
    ("probe_mean", "probe_mean_rand"),
    ("ft_pre_trans", "ft_pre"),
    ("probe_flat_trans", "probe_flat"),
    ("tangent_ridge", "corr_ridge"),
    ("tangent_krr", "tangent_ridge"),
)


def clean(obj):
    """Replace NaN and infinity by None so the result is strict JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def stat(point: np.ndarray, boot: np.ndarray | None) -> dict:
    """Seed mean and sd of the point value; CI from the replicate-wise seed mean."""
    seeds = np.asarray(point).reshape(-1)
    out = {
        "mean": float(seeds.mean()),
        "sd": float(seeds.std(ddof=1)) if len(seeds) > 1 else 0.0,
    }
    if boot is not None:
        lo, hi = percentile_ci(np.asarray(boot).reshape(len(seeds), -1).mean(axis=0))
        out["lo"], out["hi"] = lo, hi
    return out


def coverage_stats(pool: dict, parts_point: list, parts_boot: list) -> dict:
    """Coverage and width per level for one arm; empty when it has no intervals."""
    y, p, q = pool["y"], pool["p"], pool["q"]
    out = {}
    for tag, col in LEVEL_COLUMN.items():
        radius = q[..., col]
        if not np.isfinite(radius).all():
            return {}
        inside = (np.abs(y[None, :] - p) <= radius).astype(float)
        point = mean_bank(inside, parts_point)
        boot = mean_bank(inside, parts_boot)
        out[f"cover{tag}_macro"] = stat(point["macro"], boot["macro"])
        out[f"cover{tag}_pooled"] = stat(point["pooled"], boot["pooled"])
        width = mean_bank(2.0 * radius, parts_point)
        out[f"width{tag}"] = stat(width["macro"], None)
    return out


def arm_summary(pool: dict, parts_point: list, parts_boot: list) -> dict:
    point = metric_bank(pool["y"], pool["p"], parts_point)
    boot = metric_bank(pool["y"], pool["p"], parts_boot)
    out = {name: stat(point[name], boot[name]) for name in METRICS}
    out.update(coverage_stats(pool, parts_point, parts_boot))
    return out


def pair_summary(a: dict, b: dict, parts_point: list, parts_boot: list) -> dict:
    """Paired difference a minus b with the same resampled subjects for both."""
    pa = metric_bank(a["y"], a["p"], parts_point)
    pb = metric_bank(b["y"], b["p"], parts_point)
    ba = metric_bank(a["y"], a["p"], parts_boot)
    bb = metric_bank(b["y"], b["p"], parts_boot)
    return {m: stat(pa[m] - pb[m], ba[m] - bb[m]) for m in METRICS}


def summarise_mode(
    folds: list[dict], n_boot: int = 1000, seed: int = 0, pairs=PAIRS
) -> dict:
    """Summary of every (arm, target, n) and of the paired differences."""
    keys = keys_of(folds)
    cache = {k: pooled(folds, *k) for k in keys}
    site = cache[keys[0]]["site"]
    parts_point = site_parts(site)
    parts_boot = site_parts(site, n_boot, np.random.default_rng(seed))
    arms: dict = {}
    for (arm, target, n), pool in cache.items():
        entry = arm_summary(pool, parts_point, parts_boot)
        arms.setdefault(target, {}).setdefault(arm, {})[str(n)] = entry
    diffs: dict = {}
    for a, b in pairs:
        for arm_a, target, n in keys:
            if arm_a != a or (b, target, n) not in cache:
                continue
            entry = pair_summary(
                cache[(a, target, n)], cache[(b, target, n)], parts_point, parts_boot
            )
            diffs.setdefault(target, {}).setdefault(f"{a} - {b}", {})[str(n)] = entry
    return clean(
        {"n_subjects": int(len(site)), "n_boot": n_boot, "arms": arms, "diffs": diffs}
    )


def fold_means(rows: list[dict]) -> list[dict]:
    """Mean over seeds of every metric for each (mode, fold, arm, target, n)."""
    groups: dict = {}
    for r in rows:
        key = (r["mode"], r["fold"], r["arm"], r["target"], r["n"])
        groups.setdefault(key, []).append(r)
    fields = ("r2", "mae", "spearman", "cover80", "cover90", "width80", "width90")
    out = []
    for (mode, fold, arm, target, n), members in sorted(groups.items()):
        row = {"mode": mode, "fold": fold, "arm": arm, "target": target, "n": n}
        row["n_seeds"] = len(members)
        for f in fields:
            vals = [m[f] for m in members if f in m]
            row[f] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out
