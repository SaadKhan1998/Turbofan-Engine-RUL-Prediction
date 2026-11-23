"""Transformer encoder regressor: one RUL value per input sequence."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine position encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, D]
        return x + self.pe[:, : x.size(1)]


class RULTransformer(nn.Module):
    """Map a window of sensor readings [B, L, n_features] to one RUL value per window [B].

    * a linear layer projects the features to ``d_model`` so the attention width
      does not depend on how many sensors are used;
    * sinusoidal positional encoding makes the order of cycles explicit;
    * the encoder output is pooled over time (``mean`` or the ``last`` step);
    * an MLP head outputs RUL in units of the training ``target_scale``.
    """

    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, pooling: str = "mean",
                 max_len: int = 1024):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if pooling not in ("mean", "last"):
            raise ValueError("pooling must be 'mean' or 'last'")
        self.config = dict(n_features=n_features, d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout, pooling=pooling, max_len=max_len)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.pooling = pooling
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos_enc(self.input_proj(x)))
        h = h.mean(dim=1) if self.pooling == "mean" else h[:, -1]
        return self.head(h).squeeze(-1)


class LSTMRegressor(nn.Module):
    """Baseline: stacked LSTM over the window; the last hidden state predicts RUL.

    The stack is built from single-layer ``nn.LSTM`` modules with ``nn.Dropout``
    between them. This is the same computation as ``nn.LSTM(num_layers=k, dropout=p)``,
    but it avoids cuDNN's built-in RNN dropout state, whose teardown crashes the
    Python process at exit (0xC0000409) with PyTorch 2.11 + CUDA 12.8 on Windows.
    """

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.config = dict(n_features=n_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.layers = nn.ModuleList(
            nn.LSTM(n_features if i == 0 else hidden_size, hidden_size, batch_first=True) for i in range(num_layers))
        self.between = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lstm in enumerate(self.layers):
            x, _ = lstm(x if i == 0 else self.between(x))
        return self.head(x[:, -1]).squeeze(-1)


class RULEnsemble(nn.Module):
    """Average the predictions of several independently seeded models."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(models)
        self.config = models[0].config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members]).mean(dim=0)


def save_checkpoint(path: str | Path, models: list[RULTransformer], **metadata) -> None:
    """Save one or more members plus everything needed to rebuild preprocessing."""
    torch.save({
        "format_version": 2,
        "model_config": models[0].config,
        "state_dicts": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models],
        **metadata,
    }, path)


def load_checkpoint(path: str | Path, map_location="cpu"):
    """Rebuild a trained model (single or ensemble) from ``best_model.pt``.

    Returns ``(model, checkpoint_dict)``; the model is in eval mode.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dicts = ckpt.get("state_dicts") or [ckpt["state_dict"]]
    models = []
    for sd in state_dicts:
        m = RULTransformer(**ckpt["model_config"])
        m.load_state_dict(sd)
        models.append(m)
    model = models[0] if len(models) == 1 else RULEnsemble(models)
    model.eval()
    return model, ckpt
