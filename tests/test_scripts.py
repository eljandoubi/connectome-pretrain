import importlib.util
import sys
from pathlib import Path

import numpy as np

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


def write_runs(directory):
    rng = np.random.default_rng(0)
    for k, (fold, n_test) in enumerate((("A", 40), ("B", 30))):
        y = {"age": rng.normal(15, 5, n_test), "fiq": rng.normal(105, 12, n_test)}
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
    assert "pooled R2" in (out / "tables.md").read_text()
    run_main(load_script("figures"), monkeypatch, "--results", str(out))
    for name in ("fig_learning_curves.png", "fig_coverage.png", "fig_pretrain.png"):
        assert 1000 < (out / name).stat().st_size < 300_000
