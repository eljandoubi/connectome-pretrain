"""Linear probes and full fine-tuning of the ROI-token transformer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from connpre.model import ModelConfig, Regressor, RoiTransformer, crop
from connpre.pretrain import lr_at

STEPS = {50: 150, 100: 200, 300: 300}


@dataclass(frozen=True)
class FinetuneConfig:
    lr: float
    steps: int
    batch: int = 32
    weight_decay: float = 0.05
    dropout: float = 0.1
    warmup_frac: float = 0.1
    head_lr_mult: float = 10.0


def load_encoder(state, cfg: ModelConfig, device) -> RoiTransformer:
    """Encoder with the given weights (random init when state is None), in eval mode."""
    enc = RoiTransformer(cfg)
    if state is not None:
        enc.load_state_dict(state)
    return enc.to(device).eval()


def cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def random_state(cfg: ModelConfig, seed: int) -> dict:
    torch.manual_seed(seed)
    return cpu_state(RoiTransformer(cfg))


def finetune(
    state,
    model_cfg: ModelConfig,
    x: torch.Tensor,
    lab_idx: np.ndarray,
    y: np.ndarray,
    cfg: FinetuneConfig,
    seed: int,
) -> Regressor:
    """Train encoder and head on standardised labels y of the subjects in lab_idx.

    state is a pretrained encoder state dict, or None for a random initialisation.
    """
    device = x.device
    amp = device.type == "cuda"
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    enc = RoiTransformer(model_cfg)
    if state is not None:
        enc.load_state_dict(state)
    enc.set_dropout(cfg.dropout)
    enc.decoder.requires_grad_(False)
    model = Regressor(enc).to(device).train()
    body = [
        p
        for name, p in model.encoder.named_parameters()
        if not name.startswith("decoder")
    ]
    groups = [
        {"params": body, "lr": cfg.lr, "base": cfg.lr},
        {
            "params": list(model.head.parameters()),
            "lr": cfg.lr * cfg.head_lr_mult,
            "base": cfg.lr * cfg.head_lr_mult,
        },
    ]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, fused=amp)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    lab = torch.as_tensor(lab_idx, device=device)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    n, batch = len(lab), min(cfg.batch, len(lab))
    warm = max(1, int(cfg.warmup_frac * cfg.steps))
    n_frames, window = x.shape[1], model_cfg.window
    order, ptr = torch.randperm(n, device=device, generator=gen), 0
    for step in range(cfg.steps):
        factor = lr_at(step, cfg.steps, warm, 1.0)
        for g in opt.param_groups:
            g["lr"] = g["base"] * factor
        if ptr + batch > n:
            order, ptr = torch.randperm(n, device=device, generator=gen), 0
        pos = order[ptr : ptr + batch]
        ptr += batch
        starts = torch.randint(
            0, n_frames - window + 1, (batch,), device=device, generator=gen
        )
        w = crop(x, lab[pos], starts, window)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = model(w)
        loss = F.mse_loss(pred.float(), target[pos])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return model.eval()


@torch.no_grad()
def predict(
    model: Regressor,
    x: torch.Tensor,
    idx: np.ndarray,
    starts: np.ndarray,
    batch: int = 128,
) -> np.ndarray:
    """Mean prediction over the windows that start at starts."""
    window = model.encoder.cfg.window
    amp = x.device.type == "cuda"
    idx_t = torch.as_tensor(idx, device=x.device)
    out = []
    for s in range(0, len(idx_t), batch):
        sub = idx_t[s : s + batch]
        acc = torch.zeros(len(sub), device=x.device)
        for st in starts:
            w = crop(x, sub, torch.full_like(sub, int(st)), window)
            with torch.autocast(
                device_type=x.device.type, dtype=torch.float16, enabled=amp
            ):
                acc += model(w).float()
        out.append(acc / len(starts))
    return torch.cat(out).cpu().numpy().astype(np.float64)


@torch.no_grad()
def embed_all(
    encoder: RoiTransformer, x: torch.Tensor, starts: np.ndarray, batch: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token embeddings averaged over windows: mean over ROIs and flattened."""
    window = encoder.cfg.window
    amp = x.device.type == "cuda"
    means, flats = [], []
    for s in range(0, len(x), batch):
        sub = torch.arange(s, min(s + batch, len(x)), device=x.device)
        acc = 0
        for st in starts:
            w = crop(x, sub, torch.full_like(sub, int(st)), window)
            with torch.autocast(
                device_type=x.device.type, dtype=torch.float16, enabled=amp
            ):
                acc = acc + encoder.encode(w).float()
        tokens = acc / len(starts)
        means.append(tokens.mean(dim=1))
        flats.append(tokens.flatten(1))
    return torch.cat(means), torch.cat(flats)


def standardised_gram(feat: torch.Tensor, ref_idx: np.ndarray) -> np.ndarray:
    """Linear Gram matrix of features standardised with the statistics of ref_idx."""
    ref = torch.as_tensor(ref_idx, device=feat.device)
    mu = feat[ref].mean(dim=0)
    sd = feat[ref].std(dim=0).clamp_min(1e-3)
    z = (feat - mu) / sd
    return (z @ z.T).double().cpu().numpy()
