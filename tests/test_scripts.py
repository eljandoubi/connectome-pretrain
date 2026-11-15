import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from connpre.data import save_cohort, synthetic_cohort
from connpre.protocol import N_ALL
from connpre.results import RunStore

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
    "ft_pre_trans",
    "ft_trans",
)
SITE_CSV = (
    "site,n,age_mean,age_sd,fiq_mean,fiq_sd,share_dx1,share_sex1,frames_min,"
    "frames_median,lag1\nA,50,15.0,5.0,105.0,12.0,0.4,0.8,116,116.0,0.78\n"
)


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_y(k, n_test):
    rng = np.random.default_rng(100 + k)
    return {"age": rng.normal(15, 5, n_test), "fiq": rng.normal(105, 12, n_test)}


def write_long(directory, folds=(("A", 40), ("B", 30))):
    rng = np.random.default_rng(5)
    for k, (fold, n_test) in enumerate(folds):
        y = fold_y(k, n_test)
        ids = np.arange(n_test)
        store = RunStore("loso", fold, ids + 100 * k, ids, [fold] * n_test, y)
        for arm, noise in (("ft_scratch_long", 6.0), ("ft_trans_long", 4.0)):
            for seed in range(2):
                pred = y["age"] + rng.normal(0, noise, n_test)
                y_cal = rng.normal(y["age"].mean(), 5.0, 30)
                cal = (y_cal, y_cal + rng.normal(0, noise, 30))
                store.add(arm, "age", 300, seed, pred, cal)
        store.save(str(directory))


def write_runs(directory):
    rng = np.random.default_rng(0)
    for k, (fold, n_test) in enumerate((("A", 40), ("B", 30))):
        y = fold_y(k, n_test)
        ids = np.arange(n_test)
        store = RunStore("loso", fold, ids + 100 * k, ids, [fold] * n_test, y)
        for arm_index, arm in enumerate(ARMS):
            for n in (50, 100, 300, N_ALL):
                if n == N_ALL and arm.startswith(("probe", "ft")):
                    continue
                for seed in range(2):
                    for target, spread in (("age", 5.0), ("fiq", 12.0)):
                        noise = spread * (1.6 - 0.1 * arm_index)
                        pred = y[target] + rng.normal(0, noise, n_test)
                        y_cal = rng.normal(y[target].mean(), spread, 30)
                        cal = (y_cal, y_cal + rng.normal(0, noise, 30))
                        store.add(arm, target, n, seed, pred, cal)
        store.extra = {
            "pretrain": {
                "history": [
                    {
                        "step": 100,
                        "train_loss": 0.8,
                        "val_ratio": 0.9,
                        "heldout_ratio": 0.95,
                        "seconds": 5.0,
                    }
                ]
            },
            "lr": {"ft_pre|age": 1e-4},
            "seconds": 12.5,
        }
        store.save(str(directory))
    (directory / "model.json").write_text('{"parameters": 1}\n')


def run_main(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_summarise_report_and_figures_scripts_run_end_to_end(tmp_path, monkeypatch):
    runs, out = tmp_path / "runs", tmp_path / "results"
    runs.mkdir()
    write_runs(runs)
    out.mkdir()
    (out / "site_table.csv").write_text(SITE_CSV)
    run_main(
        load_script("summarise"),
        monkeypatch,
        "--runs",
        str(runs),
        "--out",
        str(out),
        "--n-boot",
        "30",
    )
    for name in (
        "summary.json",
        "per_fold.csv",
        "runs_baselines.csv",
        "runs_neural.csv",
        "pretrain_curves.json",
        "learning_rates.json",
        "timing_seconds.json",
        "model.json",
    ):
        assert (out / name).exists(), name
    run_main(load_script("report"), monkeypatch, "--results", str(out))
    text = (out / "tables.md").read_text()
    assert "pooled R2" in text
    assert "conformal intervals of the fine-tunes" in text
    head = "### age: 90 percent conformal intervals, pooled coverage (width)"
    assert "fine-tune" not in text.split(head)[1].split("###")[0]
    assert "fine-tune, transductive (own rate)" in text
    diffs = json.loads((out / "summary.json").read_text())["loso"]["diffs"]["age"]
    assert {"ft_trans - ft_pre", "ft_trans - ft_pre_trans"} <= set(diffs)
    run_main(load_script("figures"), monkeypatch, "--results", str(out))
    for name in ("fig_learning_curves.png", "fig_coverage.png", "fig_pretrain.png"):
        assert 1000 < (out / name).stat().st_size < 300_000


def test_long_fine_tunes_get_their_own_summary_and_tables(tmp_path, monkeypatch):
    runs, long, out = tmp_path / "runs", tmp_path / "long", tmp_path / "results"
    for folder in (runs, long, out):
        folder.mkdir()
    write_runs(runs)
    write_long(long)
    (out / "site_table.csv").write_text(SITE_CSV)
    run_main(
        load_script("summarise"),
        monkeypatch,
        "--runs",
        str(runs),
        "--long",
        str(long),
        "--out",
        str(out),
        "--n-boot",
        "30",
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["loso_long"]["folds"] == ["A", "B"]
    assert "ft_trans_long" in summary["loso_long"]["arms"]["age"]
    assert "ft_trans_long" not in summary["loso"]["arms"]["age"]
    pair = summary["loso_long"]["diffs"]["age"]["ft_trans_long - ft_scratch_long"]
    assert pair["300"]["pooled_r2"]["mean"] > 0
    assert "ft_trans_long" in (out / "runs_neural.csv").read_text()
    run_main(load_script("report"), monkeypatch, "--results", str(out))
    text = (out / "tables.md").read_text()
    assert "Longer fine-tuning" in text and "1000 steps" in text


def test_calibration_overlap_script_counts_the_subjects_it_drops(tmp_path, monkeypatch):
    cohort = synthetic_cohort(sizes=(60, 50, 40, 40, 30), n_roi=12, t=64, seed=5)
    data, out = tmp_path / "cohort.npz", tmp_path / "overlap.json"
    save_cohort(str(data), cohort)
    run_main(
        load_script("calibration_overlap"),
        monkeypatch,
        *("--data", str(data), "--out", str(out)),
        *("--n-cal", "40", "--lr-n-lab", "20", "--min-site", "40"),
    )
    result = json.loads(out.read_text())
    assert set(result["folds"]) == {"S0", "S1", "S2", "S3"}
    runs = [r for f in result["folds"].values() for r in f["seeds"]]
    assert len(runs) == 4 * 5
    assert all(r["kept"] + r["in_labelled"] + r["in_validation"] == 40 for r in runs)
    assert 0 < result["dropped"]["min"] <= result["dropped"]["max"] < 40
    assert result["kept"]["min"] + result["dropped"]["max"] == 40


def test_run_neural_script_runs_the_chain_on_synthetic_data(tmp_path, monkeypatch):
    cohort = synthetic_cohort(sizes=(120, 90, 80, 60, 40), n_roi=12, t=64, seed=5)
    data = tmp_path / "cohort.npz"
    save_cohort(str(data), cohort)
    script = load_script("run_neural")
    common = (
        *("--data", str(data), "--out", str(tmp_path / "runs")),
        *("--ckpt", str(tmp_path / "ckpt"), "--scale", "0.01"),
    )
    run_main(script, monkeypatch, *common, "--phase", "trans")
    run_main(
        script,
        monkeypatch,
        *common,
        *("--folds", "S0", "--seeds", "0", "--sizes", "50,100"),
        *("--n-cal", "60", "--min-site", "80"),
    )
    assert (tmp_path / "ckpt" / "trans.pt").exists()
    meta = json.loads((tmp_path / "runs" / "loso-S0.json").read_text())
    arms = {r["arm"] for r in meta["rows"]}
    assert {"ft_pre", "ft_scratch", "ft_trans", "ft_pre_trans", "probe_flat"} <= arms
    assert meta["extra"]["lr_subjects"]["labelled"] == 100
    assert all(r["n_cal"] <= 60 for r in meta["rows"])
