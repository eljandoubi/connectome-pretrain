"""Draw the study figures from results/summary.json and results/pretrain_curves.json.

usage: python scripts/figures.py [--results results]
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from connpre.report import ARM_LABEL, SIZE_LABEL, SIZES  # noqa: E402

CURVE_ARMS = (
    "tangent_ridge",
    "corr_ridge",
    "probe_flat",
    "probe_flat_rand",
    "ft_scratch",
    "ft_pre",
)
COLORS = {
    "tangent_ridge": "#1f77b4",
    "corr_ridge": "#7fb3d5",
    "probe_flat": "#2ca02c",
    "probe_flat_rand": "#a5d6a7",
    "ft_scratch": "#ff7f0e",
    "ft_pre": "#d62728",
}


def learning_curves(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
    x = np.arange(len(SIZES))
    for ax, target in zip(axes, ("age", "fiq"), strict=True):
        for k, arm in enumerate(CURVE_ARMS):
            by_size = summary["arms"].get(target, {}).get(arm)
            if not by_size:
                continue
            sizes = [n for n in SIZES if n in by_size]
            pos = np.array([SIZES.index(n) for n in sizes]) + 0.06 * (k - 2.5)
            e = [by_size[n]["pooled_r2"] for n in sizes]
            mean = np.array([v["mean"] for v in e])
            lo = np.array([v["lo"] for v in e])
            hi = np.array([v["hi"] for v in e])
            ax.errorbar(
                pos,
                mean,
                yerr=[mean - lo, hi - mean],
                marker="o",
                ms=4,
                lw=1.2,
                capsize=2,
                color=COLORS[arm],
                label=ARM_LABEL[arm],
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, [SIZE_LABEL[n] for n in SIZES])
        ax.set_xlabel("labelled subjects")
        ax.set_ylabel("pooled R2, held-out site")
        ax.set_title(f"{target} (leave-one-site-out)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def coverage_figure(summary: dict, path: Path, size: str = "100") -> None:
    arms = [
        "site_const",
        "corr_ridge",
        "tangent_ridge",
        "tangent_krr",
        "probe_mean",
        "probe_flat",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, target in zip(axes, ("age", "fiq"), strict=True):
        shown = [
            a
            for a in arms
            if "cover90_pooled"
            in summary["arms"].get(target, {}).get(a, {}).get(size, {})
        ]
        x = np.arange(len(shown))
        for j, (level, shade) in enumerate((("80", "#9ecae1"), ("90", "#3182bd"))):
            vals = [
                summary["arms"][target][a][size][f"cover{level}_pooled"] for a in shown
            ]
            mean = np.array([v["mean"] for v in vals])
            lo = np.array([v["lo"] for v in vals])
            hi = np.array([v["hi"] for v in vals])
            ax.bar(
                x + 0.2 * (2 * j - 1),
                mean,
                width=0.38,
                color=shade,
                label=f"level 0.{level}",
            )
            ax.errorbar(
                x + 0.2 * (2 * j - 1),
                mean,
                yerr=[mean - lo, hi - mean],
                fmt="none",
                ecolor="k",
                lw=0.8,
                capsize=2,
            )
            ax.axhline(int(level) / 100, color=shade, ls="--", lw=1)
        ax.set_xticks(x, [textwrap.fill(ARM_LABEL[a], 11) for a in shown], fontsize=7)
        ax.set_title(f"{target}, N={SIZE_LABEL[size]}: coverage on the held-out site")
        ax.set_ylim(0.4, 1.02)
    axes[0].set_ylabel("empirical coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pretrain_figure(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for key, color, style in (
        ("val_ratio", "#1f77b4", "-"),
        ("heldout_ratio", "#d62728", "--"),
    ):
        lines = []
        for name, data in curves.items():
            if not name.startswith("loso") or key not in data["history"][0]:
                continue
            steps = [h["step"] for h in data["history"]]
            vals = [h[key] for h in data["history"]]
            ax.plot(steps, vals, color=color, ls=style, lw=0.5, alpha=0.35)
            lines.append(vals)
        if lines:
            label = (
                "unseen subjects, training sites"
                if key == "val_ratio"
                else "held-out site"
            )
            ax.plot(
                steps, np.mean(lines, axis=0), color=color, ls=style, lw=2, label=label
            )
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("masked MSE / zero-predictor MSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default=str(Path(__file__).resolve().parents[1] / "results")
    )
    args = ap.parse_args()
    res = Path(args.results)
    summary = json.loads((res / "summary.json").read_text())["loso"]
    learning_curves(summary, res / "fig_learning_curves.png")
    coverage_figure(summary, res / "fig_coverage.png")
    curves = res / "pretrain_curves.json"
    if curves.exists():
        pretrain_figure(json.loads(curves.read_text()), res / "fig_pretrain.png")
    print("wrote figures in", res)


if __name__ == "__main__":
    main()
