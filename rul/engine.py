"""Training loop, evaluation and metrics."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn as nn

from .model import RULTransformer


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (d > 0) are penalised more than early ones."""
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


def make_criterion(name: str) -> nn.Module:
    return nn.HuberLoss() if name == "huber" else nn.MSELoss()


class DeviceBatcher:
    """Mini-batch iterator over tensors that already live on the training device.

    The whole (small) CMAPSS window set is copied to the device once; each epoch
    only a permutation of indices is generated there, so there is no per-batch
    host-to-device copy and no DataLoader overhead.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, device, shuffle: bool, seed: int = 0):
        self.X = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=device)
        self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32), device=device)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.gen = torch.Generator(device=self.X.device).manual_seed(seed) if shuffle else None

    def __len__(self) -> int:
        return math.ceil(len(self.X) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.X)
        if self.shuffle:
            idx = torch.randperm(n, device=self.X.device, generator=self.gen)
            for i in range(0, n, self.batch_size):
                j = idx[i : i + self.batch_size]
                yield self.X[j], self.y[j]
        else:
            for i in range(0, n, self.batch_size):
                yield self.X[i : i + self.batch_size], self.y[i : i + self.batch_size]


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device, target_scale: float, batch_size: int = 2048,
            clip_max: float | None = None) -> np.ndarray:
    """Predict RUL in cycles for windows ``X`` [N, L, F]; clipped to [0, clip_max]."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i : i + batch_size], device=device)
        out.append(model(xb).float().cpu().numpy())
    pred = np.concatenate(out) * target_scale
    return np.clip(pred, 0, clip_max)


@torch.no_grad()
def evaluate(model, batcher: DeviceBatcher, criterion, target_scale: float):
    """Return (loss on scaled targets, RMSE in cycles, MAE in cycles)."""
    model.eval()
    total, preds = 0.0, []
    for xb, yb in batcher:
        out = model(xb)
        total += criterion(out, yb).item() * len(xb)
        preds.append(out)
    p = torch.cat(preds) * target_scale
    t = batcher.y * target_scale
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
    mae = torch.mean(torch.abs(p - t)).item()
    return total / len(batcher.X), rmse, mae


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_model(data: dict, model_kwargs: dict, lr: float, batch_size: int, epochs: int, patience: int,
                weight_decay: float, loss: str, device, seed: int, target_scale: float,
                trial=None, verbose: bool = True, model_cls: type[nn.Module] = RULTransformer):
    """Train one model with AdamW, warmup + cosine LR and early stopping.

    ``data`` holds numpy arrays X_train, y_train, X_val, y_val (targets in cycles).
    The weights from the epoch with the lowest validation loss are restored.
    If an Optuna ``trial`` is given, validation loss is reported each epoch for pruning.

    Returns (best_model, history list[dict], best_epoch, best_val_loss).
    """
    set_seed(seed)
    train_b = DeviceBatcher(data["X_train"], data["y_train"] / target_scale, batch_size, device, True, seed)
    val_b = DeviceBatcher(data["X_val"], data["y_val"] / target_scale, 4096, device, False)

    model = model_cls(**model_kwargs).to(device)
    criterion = make_criterion(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(train_b)
    scheduler = cosine_with_warmup(optimizer, warmup_steps=min(len(train_b), steps // 10 + 1), total_steps=steps)

    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        run = torch.zeros((), device=device)
        for xb, yb in train_b:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = criterion(model(xb), yb)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            run += batch_loss.detach() * len(xb)
        train_loss = run.item() / len(train_b.X)
        val_loss, val_rmse, val_mae = evaluate(model, val_b, criterion, target_scale)

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_rmse=val_rmse,
                            val_mae=val_mae, lr=optimizer.param_groups[0]["lr"], best=improved))
        if verbose:
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.5f} | val_loss {val_loss:.5f} "
                  f"| val_RMSE {val_rmse:6.2f} | val_MAE {val_mae:6.2f}{'  *' if improved else ''}", flush=True)
        if trial is not None:
            import optuna

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad_epochs >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, best_loss
