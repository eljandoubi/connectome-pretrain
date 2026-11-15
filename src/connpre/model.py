"""ROI-token transformer: each ROI is one token holding its window of frames."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    n_roi: int = 200
    window: int = 64
    dim: int = 128
    depth: int = 5
    heads: int = 4
    mlp: int = 512


class Block(nn.Module):
    """Pre-norm self-attention block with a GELU MLP."""

    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp)
        self.fc2 = nn.Linear(mlp, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v)
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm2(x)))))


class RoiTransformer(nn.Module):
    """Encoder over ROI tokens with a linear decoder for masked reconstruction."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.window, cfg.dim)
        self.roi = nn.Parameter(torch.empty(cfg.n_roi, cfg.dim))
        self.mask_token = nn.Parameter(torch.empty(cfg.dim))
        nn.init.trunc_normal_(self.roi, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.heads, cfg.mlp) for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.decoder = nn.Linear(cfg.dim, cfg.window)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Tokens (batch, roi, dim) from windows (batch, roi, window)."""
        h = self.embed(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token.to(h.dtype), h)
        h = h + self.roi.to(h.dtype)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x, mask))

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p


class Regressor(nn.Module):
    """Encoder, mean over ROI tokens, layer norm and a linear head."""

    def __init__(self, encoder: RoiTransformer):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.cfg.dim), nn.Linear(encoder.cfg.dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.encode(x).mean(dim=1)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def random_mask(
    batch: int,
    n_roi: int,
    ratio: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean (batch, n_roi) with exactly round(ratio * n_roi) True per row."""
    k = round(ratio * n_roi)
    scores = torch.rand(batch, n_roi, device=device, generator=generator)
    chosen = scores.argsort(dim=1)[:, :k]
    mask = torch.zeros(batch, n_roi, dtype=torch.bool, device=device)
    return mask.scatter_(1, chosen, True)


def crop(
    x: torch.Tensor, idx: torch.Tensor, starts: torch.Tensor, length: int
) -> torch.Tensor:
    """Windows (batch, roi, length) of subjects idx from starts; x is (n, T, roi)."""
    frames = starts[:, None] + torch.arange(length, device=x.device)[None, :]
    return x[idx[:, None], frames].transpose(1, 2)


def masked_loss(model: RoiTransformer, windows: torch.Tensor, mask: torch.Tensor):
    """Mean squared reconstruction error over the masked tokens only."""
    pred = model(windows, mask)
    err = ((pred.float() - windows.float()) ** 2).mean(dim=-1)
    return (err * mask).sum() / mask.sum()
