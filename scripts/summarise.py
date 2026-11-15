"""Merge run folders into results/: run CSVs, per-fold means, summary and curves.

usage: python scripts/summarise.py --runs work/runs_bl work/runs_nn --out results
"""

import argparse
import json
import shutil
from pathlib import Path

from connpre.results import load_merged, write_rows_csv
from connpre.summary import (
    complete_folds,
    fold_means,
    site_wins,
    summarise_mode,
)

BASELINE_ARMS = {
    "corr_ridge",
    "tangent_ridge",
    "tangent_krr",
    "site_const",
    "site_oracle",
    "site_mean",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--long", default=None, help="run folder of the longer fine-tunes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary, rows, curves, lrs, timing = {}, [], {}, {}, {}
    held_out = None
    for mode in ("loso", "insite"):
        folds = load_merged(args.runs, mode)
        if not folds:
            continue
        kept = complete_folds(folds)
        if len(kept) < len(folds):
            names = sorted({f["fold"] for f in folds} - {f["fold"] for f in kept})
            print(mode, "folds without every arm are left out:", names)
        folds = kept
        if mode == "loso":
            held_out = {f["fold"] for f in folds}
        for f in folds:
            rows += f["meta"]["rows"]
            extra = f["meta"]["extra"]
            if "pretrain" in extra:
                curves[f"{mode}-{f['fold']}"] = extra["pretrain"]
            if "lr" in extra:
                lrs[f"{mode}-{f['fold']}"] = extra["lr"]
            if "seconds" in extra:
                timing[f"{mode}-{f['fold']}"] = extra["seconds"]
        only = held_out if mode == "insite" else None
        summary[mode] = summarise_mode(
            folds, args.n_boot, args.seed, restrict_sites=only
        )
        print(mode, "folds", len(folds), "subjects", summary[mode]["n_subjects"])
        summary[mode]["folds"] = sorted(f["fold"] for f in folds)

    if args.long:
        folds = complete_folds(load_merged([*args.runs, args.long], "loso"))
        long_rows = [
            r for f in folds for r in f["meta"]["rows"] if r["arm"].endswith("_long")
        ]
        if long_rows:
            summary["loso_long"] = summarise_mode(folds, args.n_boot, args.seed)
            summary["loso_long"]["folds"] = sorted(f["fold"] for f in folds)
            long_means = fold_means([r for f in folds for r in f["meta"]["rows"]])
            summary["loso_long"]["site_wins"] = {
                "r2": site_wins(long_means, metric="r2"),
                "mae": site_wins(long_means, metric="mae"),
            }
            rows += long_rows
            print("loso_long folds", len(folds), "rows", len(long_rows))

    base = [r for r in rows if r["arm"] in BASELINE_ARMS]
    neural = [r for r in rows if r["arm"] not in BASELINE_ARMS]
    write_rows_csv(base, str(out / "runs_baselines.csv"))
    if neural:
        write_rows_csv(neural, str(out / "runs_neural.csv"))
    means = fold_means(rows)
    write_rows_csv(means, str(out / "per_fold.csv"))
    if "loso" in summary:
        summary["loso"]["site_wins"] = {
            "r2": site_wins(means, metric="r2"),
            "mae": site_wins(means, metric="mae"),
        }
    (out / "summary.json").write_text(json.dumps(summary) + "\n")
    (out / "pretrain_curves.json").write_text(json.dumps(curves) + "\n")
    (out / "learning_rates.json").write_text(json.dumps(lrs, indent=1) + "\n")
    (out / "timing_seconds.json").write_text(json.dumps(timing, indent=1) + "\n")
    for directory in args.runs:
        for name in ("model.json", "pretrain-trans.json"):
            if (Path(directory) / name).exists():
                shutil.copy(Path(directory) / name, out / name)
    print("wrote", out)


if __name__ == "__main__":
    main()
