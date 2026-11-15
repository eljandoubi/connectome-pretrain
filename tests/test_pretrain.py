import numpy as np
import pytest
import torch

from connpre.data import synthetic_cohort
from connpre.model import ModelConfig
from connpre.pretrain import (
    PretrainConfig,
    eval_starts,
    lr_at,
    pretrain,
    reconstruction_ratio,
)

SMALL = ModelConfig(n_roi=12, window=24, dim=32, depth=2, heads=4, mlp=64)


def small_x(n=40):
    cohort = synthetic_cohort(sizes=(n // 2, n // 2), n_roi=12, t=64, seed=1)
    return torch.as_tensor(cohort.x)


def test_learning_rate_warms_up_then_decays_to_a_tenth():
    assert lr_at(0, 1000, 100, 1.0) == pytest.approx(0.01)
    assert lr_at(99, 1000, 100, 1.0) == pytest.approx(1.0)
    assert lr_at(999, 1000, 100, 1.0) == pytest.approx(0.1, abs=1e-3)
    values = [lr_at(s, 1000, 100, 1.0) for s in range(100, 1000, 100)]
    assert values == sorted(values, reverse=True)


def test_eval_starts_cover_the_series_and_are_unique():
    starts = eval_starts(116, 64, 4)
    assert starts[0] == 0 and starts[-1] == 52
    assert len(set(starts.tolist())) == len(starts)
    assert eval_starts(64, 64, 4).tolist() == [0]


def test_pretraining_lowers_the_masked_loss_on_unseen_subjects():
    x = small_x()
    cfg = PretrainConfig(steps=120, batch=16, lr=2e-3, warmup=10, log_every=40)
    before = reconstruction_ratio(
        pretrain(x, np.arange(30), PretrainConfig(steps=1, log_every=1), SMALL, 0)[0],
        x,
        np.arange(30, 40),
        0.5,
    )
    model, history = pretrain(
        x, np.arange(30), cfg, SMALL, 0, {"val": np.arange(30, 40)}
    )
    assert len(history) == 3
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["val_ratio"] < before


def test_rows_outside_the_training_indices_are_never_read():
    x = small_x()
    train = np.arange(30)
    clean, _ = pretrain(
        x, train, PretrainConfig(steps=20, batch=8, log_every=10), SMALL, 3
    )
    poisoned = x.clone()
    poisoned[30:] = float("nan")
    model, history = pretrain(
        poisoned, train, PretrainConfig(steps=20, batch=8, log_every=10), SMALL, 3
    )
    assert all(np.isfinite(h["train_loss"]) for h in history)
    for a, b in zip(clean.parameters(), model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_same_seed_gives_the_same_history():
    x = small_x()
    cfg = PretrainConfig(steps=20, batch=8, log_every=10)
    h1 = pretrain(x, np.arange(30), cfg, SMALL, 5)[1]
    h2 = pretrain(x, np.arange(30), cfg, SMALL, 5)[1]
    assert [h["train_loss"] for h in h1] == [h["train_loss"] for h in h2]
