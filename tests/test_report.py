import importlib.util
import json
from pathlib import Path

import numpy as np

from connpre.protocol import N_ALL
from connpre.report import (
    cell,
    cohort_table,
    coverage_table,
    diff_table,
    insite_table,
    long_check_table,
    point_table,
    pretrain_table,
    score_table,
    table,
    wins_table,
)
from connpre.results import RunStore, load_mode
from connpre.summary import summarise_mode

ARMS = (
    "site_const",
    "corr_ridge",
    "tangent_ridge",
    "tangent_krr",
    "probe_mean_rand",
    "probe_flat_rand",
    "probe_mean",
    "probe_flat",
    "ft_scratch",
    "ft_pre",
)


def make_summary(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    for k, (fold, n_test) in enumerate((("A", 50), ("B", 40), ("C", 30))):
        y = {"age": rng.normal(15, 5, n_test), "fiq": rng.normal(105, 12, n_test)}
        store = RunStore(
            "loso",
            fold,
            np.arange(n_test) + 100 * k,
            np.arange(n_test),
            [fold] * n_test,
            y,
        )
        for arm_index, arm in enumerate(ARMS):
            for n in (50, 100, 300, N_ALL):
                if n == N_ALL and arm.startswith(("probe", "ft")):
                    continue
                for seed_ in range(3):
                    for target, spread in (("age", 5.0), ("fiq", 12.0)):
                        noise = (
                            spread * (1.6 - 0.1 * arm_index) * (1.2 - 0.1 * np.log10(n))
                        )
                        pred = y[target] + rng.normal(0, noise, n_test)
                        y_cal = rng.normal(y[target].mean(), spread, 40)
                        cal = (y_cal, y_cal + rng.normal(0, noise, 40))
                        store.add(arm, target, n, seed_, pred, cal)
        store.save(str(tmp_path))
    return summarise_mode(load_mode(str(tmp_path), "loso"), n_boot=60)


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cell_formats_mean_and_interval():
    assert cell({"mean": 0.123, "lo": 0.1, "hi": 0.2}) == "0.12 [0.10, 0.20]"
    assert cell({"mean": 0.5}, ci=True) == "0.50"
    assert cell(None) == "-"


def test_markdown_table_has_a_separator_row():
    out = table(["a", "b"], [["1", "2"]]).splitlines()
    assert out[1] == "|---|---|"
    assert out[2] == "| 1 | 2 |"


def test_score_and_point_tables_list_the_arms_present(tmp_path):
    summary = make_summary(tmp_path)
    text = score_table(summary, "age", "pooled_r2", ARMS)
    assert "N=50" in text and "N=all" in text
    assert "tangent + ridge" in text and "fine-tune from pretrained" in text
    assert point_table(summary, "fiq", "100", ARMS).count("\n") >= len(ARMS)


def test_difference_and_coverage_tables(tmp_path):
    summary = make_summary(tmp_path)
    assert "fine-tune from pretrained minus fine-tune from scratch" in diff_table(
        summary
    )
    cov = coverage_table(summary, "age")
    assert "(" in cov and "site mean (constant)" in cov


def test_wins_table_formats_counts():
    summary = {"site_wins": {"mae": {"age": {"ft_pre - ft_scratch": {"100": [5, 8]}}}}}
    text = wins_table(summary)
    assert "fine-tune from pretrained beats fine-tune from scratch" in text
    assert "5/8" in text and text.count("-") > 3


def test_insite_table_pairs_the_two_protocols(tmp_path):
    summary = make_summary(tmp_path)
    text = insite_table(summary, summary)
    assert text.splitlines()[0].startswith("| arm | age LOSO | age in-site")


def test_cohort_and_pretrain_tables(tmp_path):
    csv_path = tmp_path / "site_table.csv"
    csv_path.write_text(
        "site,n,age_mean,age_sd,fiq_mean,fiq_sd,share_dx1,share_sex1,frames_min,"
        "frames_median,lag1\nNYU,172,15.3,6.6,110.8,14.8,0.4,0.8,176,176.0,0.78\n"
        "SBL,4,34.0,5.0,106.5,12.2,1,1,196,196.0,0.76\n"
    )
    text = cohort_table(csv_path)
    assert "| NYU | 172 | 15.3 (6.6)" in text and text.count("yes") == 1
    curves = {
        "loso-NYU": {
            "history": [
                {
                    "train_loss": 0.8,
                    "val_ratio": 0.9,
                    "heldout_ratio": 0.95,
                    "seconds": 80,
                }
            ]
        }
    }
    assert "0.950" in pretrain_table(curves)


def test_figures_are_written_and_small(tmp_path):
    summary = make_summary(tmp_path)
    figures = load_script("figures")
    figures.learning_curves(summary, tmp_path / "a.png")
    figures.coverage_figure(summary, tmp_path / "b.png")
    steps = [{"step": s, "val_ratio": 0.9, "heldout_ratio": 0.95} for s in (100, 200)]
    figures.pretrain_figure({"loso-A": {"history": steps}}, tmp_path / "c.png")
    for name in "abc":
        size = (tmp_path / f"{name}.png").stat().st_size
        assert 1000 < size < 300_000


def test_summary_json_round_trips(tmp_path):
    summary = make_summary(tmp_path)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"loso": summary}))
    assert json.loads(path.read_text())["loso"]["n_subjects"] == 120


def test_long_check_table_lists_every_run():
    scores = {"r2": 0.0123, "mae": 4.0, "spearman": 0.456}
    check = {
        "site": "NYU",
        "n": 300,
        "target": "age",
        "runs": [
            {
                "arm": "scratch",
                "lr": 3e-4,
                "steps": 300,
                "train": scores,
                "held_out": scores,
            },
            {
                "arm": "transductive_pretrained",
                "lr": 1e-4,
                "steps": 1500,
                "train": scores,
                "held_out": scores,
            },
        ],
    }
    lines = long_check_table(check).split("\n")
    assert len(lines) == 4
    assert lines[2].startswith("| scratch | 0.0003 | 300 | 0.012 | 0.46 | 0.012 | 0.46")
    assert "transductive pretrained | 0.0001 | 1500" in lines[3]
