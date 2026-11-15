"""Compare the tangent embedding with nilearn on a subset of real subjects.

usage: python scripts/check_tangent.py --data data/abide_cc200.npz \
    --out results/tangent_check.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf

from connpre.data import load_cohort
from connpre.features import TangentSpace, ledoit_wolf_covs


def nilearn_tangent(series: list) -> np.ndarray:
    kwargs = dict(
        kind="tangent", cov_estimator=LedoitWolf(store_precision=False), vectorize=False
    )
    try:
        measure = ConnectivityMeasure(standardize=False, **kwargs)
    except TypeError:
        measure = ConnectivityMeasure(**kwargs)
    return measure.fit_transform(series)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    cohort = load_cohort(args.data)
    pick = np.sort(np.random.default_rng(0).choice(len(cohort), args.n, replace=False))
    series = cohort.x[pick].astype(np.float64)
    covs = ledoit_wolf_covs(series)
    theirs = nilearn_tangent(list(series))
    result = {"n_subjects": int(args.n), "n_roi": int(series.shape[2])}
    for tol in (1e-3, 1e-8):
        space = TangentSpace(tol=tol).fit(covs)
        mine = space.transform_matrices(covs)
        diff = mine - theirs
        rel = np.linalg.norm(diff) / np.linalg.norm(theirs)
        result[f"tol_{tol:g}"] = {
            "iterations": int(space.n_iter_),
            "max_abs_diff": float(np.abs(diff).max()),
            "relative_frobenius_diff": float(rel),
        }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
