"""Markdown tables built from results/summary.json and the other result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

SIZES = ("50", "100", "300", "1000000000")
SIZE_LABEL = {"50": "50", "100": "100", "300": "300", "1000000000": "all"}
BASELINES = ("site_const", "corr_ridge", "tangent_ridge", "tangent_krr")
NEURAL = (
    "probe_mean_rand",
    "probe_flat_rand",
    "probe_mean",
    "probe_flat",
    "ft_scratch",
    "ft_pre",
)
TRANSDUCTIVE = ("probe_mean_trans", "probe_flat_trans", "ft_pre_trans")
ARM_LABEL = {
    "site_const": "site mean (constant)",
    "site_oracle": "site mean of the held-out site (oracle)",
    "site_mean": "site mean of the labelled subjects of the site",
    "corr_ridge": "correlation + ridge",
    "tangent_ridge": "tangent + ridge",
    "tangent_krr": "tangent + RBF kernel ridge",
    "probe_mean_rand": "probe, mean token, random encoder",
    "probe_flat_rand": "probe, all tokens, random encoder",
    "probe_mean": "probe, mean token, pretrained",
    "probe_flat": "probe, all tokens, pretrained",
    "ft_scratch": "fine-tune from scratch",
    "ft_pre": "fine-tune from pretrained",
    "probe_mean_trans": "probe, mean token, transductive",
    "probe_flat_trans": "probe, all tokens, transductive",
    "ft_pre_trans": "fine-tune, transductive",
    "ft_scratch_long": "fine-tune from scratch, 1000 steps",
    "ft_trans_long": "fine-tune from transductive encoder, 1000 steps",
}
LONG_ARMS = (
    "site_const",
    "tangent_ridge",
    "ft_scratch",
    "ft_pre_trans",
    "ft_scratch_long",
    "ft_trans_long",
)


def cell(entry: dict | None, digits: int = 2, ci: bool = True) -> str:
    if not entry or entry.get("mean") is None:
        return "-"
    text = f"{entry['mean']:.{digits}f}"
    if ci and entry.get("lo") is not None:
        text += f" [{entry['lo']:.{digits}f}, {entry['hi']:.{digits}f}]"
    return text


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def arm_rows(
    summary: dict, target: str, arms, sizes, metric: str, digits: int = 2, ci=True
):
    rows = []
    for arm in arms:
        by_size = summary["arms"].get(target, {}).get(arm)
        if by_size is None:
            continue
        cells = [cell(by_size.get(n, {}).get(metric), digits, ci) for n in sizes]
        rows.append([ARM_LABEL.get(arm, arm), *cells])
    return rows


def score_table(summary: dict, target: str, metric: str, arms, sizes=SIZES, digits=2):
    header = ["arm", *[f"N={SIZE_LABEL[n]}" for n in sizes]]
    return table(header, arm_rows(summary, target, arms, sizes, metric, digits))


def point_table(summary: dict, target: str, size: str, arms) -> str:
    """Several metrics at one label size."""
    cols = ("pooled_r2", "macro_r2", "pooled_mae", "macro_spearman")
    header = ["arm", "pooled R2", "macro R2", "pooled MAE", "macro Spearman"]
    rows = []
    for arm in arms:
        e = summary["arms"].get(target, {}).get(arm, {}).get(size)
        if e:
            rows.append([ARM_LABEL.get(arm, arm), *[cell(e.get(c)) for c in cols]])
    return table(header, rows)


def diff_table(
    summary: dict, metric: str = "pooled_r2", sizes=("50", "100", "300")
) -> str:
    header = ["difference"]
    for target in ("age", "fiq"):
        header += [f"{target} N={SIZE_LABEL[n]}" for n in sizes]
    pairs = sorted({p for t in summary["diffs"].values() for p in t})
    rows = []
    for pair in pairs:
        a, b = pair.split(" - ")
        row = [f"{ARM_LABEL.get(a, a)} minus {ARM_LABEL.get(b, b)}"]
        for target in ("age", "fiq"):
            for n in sizes:
                e = (
                    summary["diffs"]
                    .get(target, {})
                    .get(pair, {})
                    .get(n, {})
                    .get(metric)
                )
                row.append(cell(e))
        rows.append(row)
    return table(header, rows)


def wins_table(summary: dict, metric: str = "mae", sizes=("50", "100", "300")) -> str:
    """Held-out sites won by the first arm of each pair."""
    wins = summary.get("site_wins", {}).get(metric, {})
    header = ["pair"]
    for target in ("age", "fiq"):
        header += [f"{target} N={SIZE_LABEL[n]}" for n in sizes]
    pairs = sorted({p for t in wins.values() for p in t})
    rows = []
    for pair in pairs:
        a, b = pair.split(" - ")
        row = [f"{ARM_LABEL.get(a, a)} beats {ARM_LABEL.get(b, b)}"]
        for target in ("age", "fiq"):
            for n in sizes:
                value = wins.get(target, {}).get(pair, {}).get(n)
                row.append("-" if value is None else f"{value[0]}/{value[1]}")
        rows.append(row)
    return table(header, rows)


def long_diff_table(summary: dict, size: str = "300") -> str:
    """Paired differences of the longer fine-tunes (age), several metrics."""
    cols = ("pooled_r2", "macro_r2", "pooled_mae", "macro_spearman")
    header = [
        "difference (age)",
        "pooled R2",
        "macro R2",
        "pooled MAE",
        "macro Spearman",
    ]
    pairs = [
        p
        for p in summary.get("diffs", {}).get("age", {})
        if p.split(" - ")[0].endswith("_long")
    ]
    rows = []
    for pair in sorted(pairs):
        a, b = pair.split(" - ")
        entry = summary["diffs"]["age"][pair].get(size, {})
        rows.append(
            [
                f"{ARM_LABEL.get(a, a)} minus {ARM_LABEL.get(b, b)}",
                *[cell(entry.get(c)) for c in cols],
            ]
        )
    return table(header, rows)


def long_check_table(check: dict) -> str:
    """The one-fold check of fine-tuning length, from long_finetune.json."""
    header = [
        "init",
        "lr",
        "steps",
        "train R2",
        "train Spearman",
        "held-out R2",
        "held-out Spearman",
    ]
    rows = []
    for r in check["runs"]:
        rows.append(
            [
                r["arm"].replace("_", " "),
                f"{r['lr']:g}",
                str(r["steps"]),
                f"{r['train']['r2']:.3f}",
                f"{r['train']['spearman']:.2f}",
                f"{r['held_out']['r2']:.3f}",
                f"{r['held_out']['spearman']:.2f}",
            ]
        )
    return table(header, rows)


def coverage_table(
    summary: dict, target: str, level: str = "90", sizes=SIZES, arms=None
) -> str:
    arms = arms or [*BASELINES, *NEURAL, *TRANSDUCTIVE]
    header = ["arm", *[f"N={SIZE_LABEL[n]}" for n in sizes]]
    rows = []
    for arm in arms:
        by_size = summary["arms"].get(target, {}).get(arm)
        if not by_size:
            continue
        cells = []
        for n in sizes:
            e = by_size.get(n, {})
            if f"cover{level}_pooled" not in e:
                cells.append("-")
                continue
            cover = cell(e[f"cover{level}_pooled"], 2, ci=False)
            cells.append(f"{cover} ({e[f'width{level}']['mean']:.1f})")
        rows.append([ARM_LABEL.get(arm, arm), *cells])
    return table(header, rows)


def insite_table(loso: dict, insite: dict, size: str = "100") -> str:
    header = ["arm", "age LOSO", "age in-site", "fiq LOSO", "fiq in-site"]
    arms = [*BASELINES, "site_mean", *NEURAL]
    rows = []
    for arm in arms:
        row = [ARM_LABEL.get(arm, arm)]
        for target in ("age", "fiq"):
            for s in (loso, insite):
                e = (
                    s["arms"]
                    .get(target, {})
                    .get(arm, {})
                    .get(size, {})
                    .get("pooled_r2")
                )
                row.append(cell(e))
        if any(c != "-" for c in row[1:]):
            rows.append(row)
    return table(header, rows)


def cohort_table(site_csv: Path, min_site: int = 40) -> str:
    rows = list(csv.DictReader(open(site_csv)))
    header = [
        "site",
        "n",
        "age mean (sd)",
        "FIQ mean (sd)",
        "frames in file",
        "lag-1 autocorr",
        "held out",
    ]
    out = []
    for r in rows:
        held = "yes" if int(r["n"]) >= min_site else ""
        out.append(
            [
                r["site"],
                r["n"],
                f"{float(r['age_mean']):.1f} ({float(r['age_sd']):.1f})",
                f"{float(r['fiq_mean']):.1f} ({float(r['fiq_sd']):.1f})",
                f"{float(r['frames_median']):.0f}",
                f"{float(r['lag1']):.2f}",
                held,
            ]
        )
    return table(header, out)


def pretrain_table(curves: dict) -> str:
    header = ["fold", "train loss", "val ratio", "held-out ratio", "seconds"]
    rows = []
    for name in sorted(curves):
        last = curves[name]["history"][-1]
        rows.append(
            [
                name,
                f"{last['train_loss']:.3f}",
                f"{last['val_ratio']:.3f}",
                f"{last['heldout_ratio']:.3f}",
                f"{last['seconds']:.0f}",
            ]
        )
    return table(header, rows)


def load_summary(path: str) -> dict:
    return json.loads(Path(path).read_text())
