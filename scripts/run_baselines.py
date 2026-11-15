"""Run the baseline arms on every fold and save the runs.

usage: python scripts/run_baselines.py --data data/abide_cc200.npz --out work/runs
"""

import argparse
import time
from pathlib import Path

from connpre.baselines import fold_baselines
from connpre.data import load_cohort
from connpre.features import correlation_vectors, ledoit_wolf_covs
from connpre.protocol import N_ALL, insite_folds, loso_folds
from connpre.settings import INSITE_SPLITS, LABEL_SIZES, MIN_SITE, N_CAL, SEEDS


def parse_sizes(text: str) -> tuple:
    return tuple(N_ALL if s == "all" else int(s) for s in text.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="loso,insite")
    ap.add_argument("--folds", default="", help="comma list of fold names, default all")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument(
        "--sizes",
        default=",".join("all" if n == N_ALL else str(n) for n in LABEL_SIZES),
    )
    ap.add_argument("--min-site", type=int, default=MIN_SITE)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    args = ap.parse_args()

    cohort = load_cohort(args.data)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    sizes = parse_sizes(args.sizes)
    keep = set(filter(None, args.folds.split(",")))
    start = time.time()
    covs = ledoit_wolf_covs(cohort.x)
    corr = correlation_vectors(cohort.x)
    print("features", cohort.x.shape, round(time.time() - start), "s")
    for mode in args.modes.split(","):
        if mode == "loso":
            folds = loso_folds(cohort.site, args.min_site)
        else:
            folds = insite_folds(cohort.site, INSITE_SPLITS, seed=0)
        for fold in folds:
            if keep and fold.name not in keep:
                continue
            if (Path(args.out) / f"{mode}-{fold.name}.json").exists():
                print("skip", mode, fold.name)
                continue
            store = fold_baselines(
                cohort, fold, covs, corr, mode, seeds, sizes, args.n_cal
            )
            store.save(args.out)
            print(
                mode,
                fold.name,
                "test",
                len(fold.test),
                "rows",
                len(store.rows),
                round(time.time() - start),
                "s",
                flush=True,
            )


if __name__ == "__main__":
    main()
