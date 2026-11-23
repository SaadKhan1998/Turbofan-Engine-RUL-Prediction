"""Static figures for training curves and test-set evaluation (PNG, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loss_curve(history: pd.DataFrame, col: str, title: str, path: Path) -> None:
    """One line per ensemble member (same hue); a dot marks each member's kept checkpoint."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    members = sorted(history["member"].unique())
    for m in members:
        h = history[history["member"] == m]
        ax.plot(h["epoch"], h[col], color=BLUE, alpha=0.55 if len(members) > 1 else 1.0, lw=1.6)
        best = h.loc[h["val_loss"].idxmin()]
        ax.plot(best["epoch"], best[col], "o", ms=7, color=BLUE, mec=SURFACE, mew=2)
    n = len(members)
    ax.set_title(title + (f" ({n} ensemble members; dot = kept checkpoint)" if n > 1 else " (dot = kept checkpoint)"))
    ax.set(xlabel="Epoch", ylabel=f"{col.replace('_', ' ')} (scaled RUL)")
    _save(fig, path)


def pred_vs_true(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    hi = float(max(np.max(y_true), np.max(y_pred))) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", label="perfect prediction")
    ax.scatter(y_true, y_pred, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="test engine")
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="True RUL (cycles)", ylabel="Predicted RUL (cycles)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def engines_sorted(y_true, y_pred, path: Path, title: str) -> None:
    order = np.argsort(y_true)
    x = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, np.asarray(y_true)[order], color=BLUE, label="true RUL")
    ax.plot(x, np.asarray(y_pred)[order], "o", ms=5, color=ORANGE, mec=SURFACE, mew=1, label="predicted RUL")
    ax.set(xlabel="Test engines, sorted by true RUL", ylabel="RUL (cycles)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def model_comparison(results: pd.DataFrame, path: Path, title: str, highlight: str) -> None:
    """Horizontal bars of test RMSE; the highlighted model is blue, the others neutral."""
    r = results.sort_values("test_rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(r) + 1.2))
    colors = [BLUE if r_.startswith(highlight) else AXIS for r_ in r["model"]]
    bars = ax.barh(r["model"], r["test_rmse"], color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, r["test_rmse"], strict=True):
        ax.text(b.get_width() + r["test_rmse"].max() * 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, r["test_rmse"].max() * 1.12)
    ax.set_xlabel("Test RMSE (cycles)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title(title)
    _save(fig, path)


def trajectories(curves: list[dict], path: Path, title: str) -> None:
    """Small multiples: predicted vs true RUL over the observed life of selected test engines."""
    n = len(curves)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows), sharey=True, squeeze=False)
    for ax, c in zip(axes.flat[:n], curves, strict=True):
        ax.plot(c["cycle"], c["true"], color=BLUE, label="true RUL (capped)")
        ax.plot(c["cycle"], c["pred"], color=ORANGE, label="predicted RUL")
        ax.set_title(f"Engine {c['engine_id']}", fontsize=10)
        ax.set_xlabel("Cycle")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("RUL (cycles)")
    axes.flat[0].legend(loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, path)
