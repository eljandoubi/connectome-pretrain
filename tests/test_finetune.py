import numpy as np
import pytest
import torch

from connpre.data import synthetic_cohort
from connpre.finetune import (
    FinetuneConfig,
    cpu_state,
    embed_all,
    finetune,
    load_encoder,
    predict,
    random_state,
    standardised_gram,
)
from connpre.model import ModelConfig
from connpre.pretrain import eval_starts
from connpre.ridge import ridge_cv_predict

SMALL = ModelConfig(n_roi=12, window=24, dim=32, depth=2, heads=4, mlp=64)


@pytest.fixture(scope="module")
def data():
    cohort = synthetic_cohort(sizes=(30, 30), n_roi=12, t=64, seed=2)
    return cohort, torch.as_tensor(cohort.x)


def zscore(v):
    return (v - v.mean()) / v.std()


def test_finetune_fits_a_function_of_the_input_better_than_a_constant():
    cohort = synthetic_cohort(sizes=(30, 30), n_roi=12, t=24, seed=2)
    x = torch.as_tensor(cohort.x)
    lab = np.arange(48)
    y = zscore(cohort.x[:, 0, :].mean(axis=1))[lab]
    cfg = FinetuneConfig(lr=3e-3, steps=150, batch=16, dropout=0.0)
    model = finetune(None, SMALL, x, lab, y, cfg, seed=0)
    pred = predict(model, x, lab, np.array([0]))
    assert pred.shape == (48,)
    assert np.mean((pred - y) ** 2) < 0.5 * np.mean((y - y.mean()) ** 2)


def test_pretrained_state_is_the_starting_point_and_scratch_is_not(data):
    _, x = data
    state = random_state(SMALL, seed=11)
    cfg = FinetuneConfig(lr=1e-9, steps=2, batch=8)
    lab, y = np.arange(16), np.zeros(16)
    same = finetune(state, SMALL, x, lab, y, cfg, seed=0)
    other = finetune(None, SMALL, x, lab, y, cfg, seed=0)
    assert torch.allclose(same.encoder.embed.weight, state["embed.weight"], atol=1e-6)
    assert not torch.allclose(
        other.encoder.embed.weight, state["embed.weight"], atol=1e-3
    )


def test_finetune_moves_the_encoder_weights(data):
    cohort, x = data
    state = random_state(SMALL, seed=1)
    lab = np.arange(30)
    cfg = FinetuneConfig(lr=1e-2, steps=20, batch=8)
    model = finetune(state, SMALL, x, lab, zscore(cohort.age)[lab], cfg, seed=0)
    assert not torch.allclose(
        model.encoder.embed.weight, state["embed.weight"], atol=1e-4
    )
    assert torch.equal(model.encoder.decoder.weight, state["decoder.weight"])


def test_embeddings_have_the_documented_shapes_and_do_not_change_weights(data):
    _, x = data
    enc = load_encoder(random_state(SMALL, seed=3), SMALL, "cpu")
    before = cpu_state(enc)
    starts = eval_starts(x.shape[1], SMALL.window, 3)
    mean, flat = embed_all(enc, x, starts, batch=16)
    assert mean.shape == (60, 32) and flat.shape == (60, 12 * 32)
    assert all(torch.equal(before[k], v) for k, v in cpu_state(enc).items())
    assert torch.allclose(flat.view(60, 12, 32).mean(dim=1), mean, atol=1e-5)


def test_probe_is_a_ridge_on_the_standardised_embedding_gram(data):
    cohort, x = data
    enc = load_encoder(random_state(SMALL, seed=4), SMALL, "cpu")
    starts = eval_starts(x.shape[1], SMALL.window, 3)
    mean, _ = embed_all(enc, x, starts)
    gram = standardised_gram(mean, np.arange(60))
    z = ((mean - mean.mean(0)) / mean.std(0).clamp_min(1e-3)).double().numpy()
    assert np.allclose(gram, z @ z.T, atol=1e-6)
    y = np.stack([cohort.age, cohort.fiq], axis=1)
    pred, _ = ridge_cv_predict(
        gram, y, np.arange(40), np.arange(40, 60), cohort.site[:40]
    )
    assert pred.shape == (20, 2) and np.isfinite(pred).all()


def test_two_runs_with_the_same_seed_agree(data):
    cohort, x = data
    lab = np.arange(30)
    y = zscore(cohort.age)[lab]
    cfg = FinetuneConfig(lr=1e-3, steps=10, batch=8)
    a = finetune(None, SMALL, x, lab, y, cfg, seed=5)
    b = finetune(None, SMALL, x, lab, y, cfg, seed=5)
    assert torch.equal(a.head[1].weight, b.head[1].weight)
