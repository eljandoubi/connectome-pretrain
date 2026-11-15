"""Print the markdown tables of the study from results/ and write results/tables.md.

usage: python scripts/report.py [--results results]
"""

import argparse
import json
from pathlib import Path

from connpre.report import (
    BASELINES,
    NEURAL,
    TRANSDUCTIVE,
    cohort_table,
    coverage_table,
    diff_table,
    insite_table,
    point_table,
    pretrain_table,
    score_table,
    wins_table,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default=str(Path(__file__).resolve().parents[1] / "results")
    )
    args = ap.parse_args()
    res = Path(args.results)
    summary = json.loads((res / "summary.json").read_text())
    loso = summary["loso"]
    parts = []

    def add(title, body):
        parts.append(f"### {title}\n\n{body}\n")

    add("Cohort by site", cohort_table(res / "site_table.csv"))
    arms = [*BASELINES, *NEURAL]
    for target in ("age", "fiq"):
        add(
            f"{target}: pooled R2 by label count (leave-one-site-out)",
            score_table(loso, target, "pooled_r2", arms),
        )
        add(
            f"{target}: macro R2 by label count",
            score_table(loso, target, "macro_r2", arms),
        )
        add(
            f"{target}: all metrics at N=100",
            point_table(loso, target, "100", [*arms, "site_oracle"]),
        )
        add(
            f"{target}: transductive encoders, pooled R2",
            score_table(
                loso,
                target,
                "pooled_r2",
                [*TRANSDUCTIVE, "ft_pre"],
                sizes=("50", "100", "300"),
            ),
        )
        add(
            f"{target}: 90 percent conformal intervals, pooled coverage (width)",
            coverage_table(loso, target),
        )
    add("Paired differences, pooled R2", diff_table(loso, "pooled_r2"))
    add("Paired differences, macro R2", diff_table(loso, "macro_r2"))
    add("Sites won on MAE (seed mean per site)", wins_table(loso, "mae"))
    if "insite" in summary:
        add(
            "In-site reference versus leave-one-site-out, N=100, pooled R2",
            insite_table(loso, summary["insite"]),
        )
    curves = res / "pretrain_curves.json"
    if curves.exists():
        data = {
            k: v
            for k, v in json.loads(curves.read_text()).items()
            if k.startswith("loso")
        }
        if data:
            add("Masked pretraining, ratio to the zero predictor", pretrain_table(data))
    text = "\n".join(parts)
    (res / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
