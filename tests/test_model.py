import torch

from connpre.model import (
    ModelConfig,
    Regressor,
    RoiTransformer,
    count_params,
    crop,
    masked_loss,
    random_mask,
)

SMALL = ModelConfig(n_roi=12, window=16, dim=32, depth=2, heads=4, mlp=64)


def test_default_model_has_about_one_million_parameters():
    n = count_params(RoiTransformer(ModelConfig()))
    assert 0.9e6 < n < 1.2e6


def test_random_mask_has_the_exact_count_and_differs_between_rows():
    gen = torch.Generator().manual_seed(0)
    mask = random_mask(8, 200, 0.5, "cpu", gen)
    assert mask.dtype == torch.bool and mask.shape == (8, 200)
    assert bool((mask.sum(dim=1) == 100).all())
    assert not torch.equal(mask[0], mask[1])


def test_crop_returns_the_requested_frames_as_roi_by_time():
    x = torch.arange(3 * 10 * 4, dtype=torch.float32).reshape(3, 10, 4)
    w = crop(x, torch.tensor([2, 0]), torch.tensor([3, 5]), 4)
    assert w.shape == (2, 4, 4)
    assert torch.equal(w[0], x[2, 3:7].T)
    assert torch.equal(w[1], x[0, 5:9].T)


def test_encoder_and_reconstruction_shapes():
    model = RoiTransformer(SMALL).eval()
    x = torch.randn(3, 12, 16)
    assert model.encode(x).shape == (3, 12, 32)
    mask = random_mask(3, 12, 0.5, "cpu")
    assert model(x, mask).shape == (3, 12, 16)


def test_loss_counts_only_masked_tokens():
    class Stub:
        def __call__(self, w, mask):
            out = w.clone()
            out[mask] += 1.0
            out[~mask] += 100.0
            return out

    w = torch.randn(4, 12, 16)
    mask = random_mask(4, 12, 0.5, "cpu")
    assert float(masked_loss(Stub(), w, mask)) == 1.0


def test_masked_windows_cannot_reach_the_output():
    model = RoiTransformer(SMALL).eval()
    x1 = torch.randn(2, 12, 16)
    mask = random_mask(2, 12, 0.5, "cpu")
    x2 = x1.clone()
    x2[mask] = torch.randn(int(mask.sum()), 16) * 10
    with torch.no_grad():
        assert torch.allclose(model(x1, mask), model(x2, mask), atol=1e-5)


def test_regressor_gives_one_value_per_subject_and_trains_the_encoder():
    enc = RoiTransformer(SMALL)
    enc.decoder.requires_grad_(False)
    reg = Regressor(enc)
    out = reg(torch.randn(5, 12, 16))
    assert out.shape == (5,)
    out.sum().backward()
    assert enc.embed.weight.grad is not None
    assert reg.head[1].weight.grad is not None
    assert enc.decoder.weight.grad is None


def test_dropout_is_off_by_default_and_switchable():
    model = RoiTransformer(SMALL)
    x = torch.randn(2, 12, 16)
    model.train()
    assert torch.equal(model.encode(x), model.encode(x))
    model.set_dropout(0.5)
    assert not torch.equal(model.encode(x), model.encode(x))
    model.eval()
    assert torch.equal(model.encode(x), model.encode(x))
