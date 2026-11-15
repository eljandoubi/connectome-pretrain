"""Count the calibration subjects that also chose the fine-tuning learning rates.

usage: python scripts/calibration_overlap.py --data D \
    --out results/calibration_overlap.json

The first protocol drew the calibration subjects from every training subject, so some
of them were among the labelled or validation subjects of the learning-rate choice.
The fine-tune intervals now calibrate on the others only; this script counts both.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from connpre.data import load_cohort
from connpre.neural import calibration_without_tuning, fold_seed, lr_selection_sets
from connpre.protocol import draw_sets, loso_folds
from connpre.settings import MIN_SITE, N_CAL, SEEDS


def spread(values: list) -> dict:
    return {
        "min": int(min(values)),
        "mean": float(np.mean(values)),
        "max": int(max(values)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    ap.add_argument("--lr-n-lab", type=int, default=100)
    ap.add_argument("--min-site", type=int, default=MIN_SITE)
    args = ap.parse_args()

    site = load_cohort(args.data).site
    folds, dropped, kept = {}, [], []
    for fold in loso_folds(site, args.min_site):
        lab, val = lr_selection_sets(
            fold.train, site, fold.name, True, fold_seed(fold.name), args.lr_n_lab
        )
        runs = []
        for seed in SEEDS:
            cal = draw_sets(fold.train, site, args.n_cal, seed, fold.name).cal
            left = calibration_without_tuning(cal, lab, val)
            runs.append(
                {
                    "seed": seed,
                    "in_labelled": int(np.isin(cal, lab).sum()),
                    "in_validation": int(np.isin(cal, val).sum()),
                    "kept": int(len(left)),
                }
            )
            dropped.append(len(cal) - len(left))
            kept.append(len(left))
        folds[fold.name] = {
            "n_train": int(len(fold.train)),
            "validation_site": str(site[val[0]]),
            "n_validation": int(len(val)),
            "seeds": runs,
        }
    out = {
        "n_cal": args.n_cal,
        "lr_n_labelled": args.lr_n_lab,
        "dropped": spread(dropped),
        "kept": spread(kept),
        "folds": folds,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("dropped", out["dropped"], "kept", out["kept"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
