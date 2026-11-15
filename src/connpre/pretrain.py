"""Masked reconstruction pretraining on training-site subjects only."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from connpre.model import ModelConfig, RoiTransformer, crop, masked_loss, random_mask


@dataclass(frozen=True)
class PretrainConfig:
    steps: int = 3000
    batch: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup: int = 150
    mask_ratio: float = 0.5
    log_every: int = 100


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warmup then cosine decay to a tenth of the base rate."""
    if step < warmup:
        return base * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def eval_starts(n_frames: int, window: int, k: int) -> np.ndarray:
    """k evenly spaced window starts covering the series."""
    return np.unique(np.linspace(0, n_frames - window, k).round().astype(np.int64))


@torch.no_grad()
def reconstruction_ratio(
    model: RoiTransformer,
    x: torch.Tensor,
    idx: np.ndarray,
    ratio: float,
    seed: int = 0,
) -> float:
    """Masked MSE divided by the MSE of predicting zeros, on fixed windows and masks."""
    cfg = model.cfg
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    sub = torch.as_tensor(idx, device=x.device)
    starts = torch.full_like(sub, (x.shape[1] - cfg.window) // 2)
    w = crop(x, sub, starts, cfg.window)
    mask = random_mask(len(sub), cfg.n_roi, ratio, x.device, gen)
    amp = x.device.type == "cuda"
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
        loss = masked_loss(model, w, mask)
    zero = ((w.float() ** 2).mean(dim=-1) * mask).sum() / mask.sum()
    model.train(was_training)
    return float(loss / zero)


def pretrain(
    x: torch.Tensor,
    train_idx: np.ndarray,
    cfg: PretrainConfig,
    model_cfg: ModelConfig,
    seed: int,
    monitor: dict[str, np.ndarray] | None = None,
) -> tuple[RoiTransformer, list[dict]]:
    """Train on windows of the subjects in train_idx; x is (n, T, roi) on the device.

    Only rows of x listed in train_idx are read for the loss. The monitor subjects
    (name to indices) are scored every log step and never used for training.
    """
    device = x.device
    amp = device.type == "cuda"
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = RoiTransformer(model_cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=amp,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pool = torch.as_tensor(train_idx, device=device)
    n_frames, window = x.shape[1], model_cfg.window
    history, running = [], []
    start = time.time()
    model.train()
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg.steps, cfg.warmup, cfg.lr)
        pick = pool[
            torch.randint(len(pool), (cfg.batch,), device=device, generator=gen)
        ]
        starts = torch.randint(
            0, n_frames - window + 1, (cfg.batch,), device=device, generator=gen
        )
        w = crop(x, pick, starts, window)
        mask = random_mask(cfg.batch, model_cfg.n_roi, cfg.mask_ratio, device, gen)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = masked_loss(model, w, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running.append(loss.detach())
        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
            entry = {
                "step": step + 1,
                "train_loss": float(torch.stack(running).mean()),
                "seconds": round(time.time() - start, 1),
            }
            for name, idx in (monitor or {}).items():
                entry[f"{name}_ratio"] = reconstruction_ratio(
                    model, x, idx, cfg.mask_ratio
                )
            history.append(entry)
            running = []
    model.eval()
    return model, history
