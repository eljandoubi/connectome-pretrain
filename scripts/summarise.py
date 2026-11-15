"""Merge run folders into results/: run CSVs, per-fold means, summary and curves.

usage: python scripts/summarise.py --runs work/runs_bl work/runs_nn --out results
"""

import argparse
import json
from pathlib import Path

from connpre.results import load_merged, write_rows_csv
from connpre.summary import fold_means, summarise_mode

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
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary, rows, curves, lrs = {}, [], {}, {}
    for mode in ("loso", "insite"):
        folds = load_merged(args.runs, mode)
        if not folds:
            continue
        for f in folds:
            rows += f["meta"]["rows"]
            extra = f["meta"]["extra"]
            if "pretrain" in extra:
                curves[f"{mode}-{f['fold']}"] = extra["pretrain"]
            if "lr" in extra:
                lrs[f"{mode}-{f['fold']}"] = extra["lr"]
        summary[mode] = summarise_mode(folds, args.n_boot, args.seed)
        print(mode, "folds", len(folds), "subjects", summary[mode]["n_subjects"])

    base = [r for r in rows if r["arm"] in BASELINE_ARMS]
    neural = [r for r in rows if r["arm"] not in BASELINE_ARMS]
    write_rows_csv(base, str(out / "runs_baselines.csv"))
    if neural:
        write_rows_csv(neural, str(out / "runs_neural.csv"))
    write_rows_csv(fold_means(rows), str(out / "per_fold.csv"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "pretrain_curves.json").write_text(json.dumps(curves) + "\n")
    (out / "learning_rates.json").write_text(json.dumps(lrs, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
