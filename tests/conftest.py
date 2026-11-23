"""Shared fixtures: a tiny synthetic dataset in the exact CMAPSS file format."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _engine_rows(eid: int, n_cycles: int, life: int, rng) -> list[list[float]]:
    rows = []
    for c in range(1, n_cycles + 1):
        wear = c / life
        settings = rng.normal(0, 0.001, 3)
        sensors = 1.0 + 5.0 * wear**2 + rng.normal(0, 0.05, 21)
        sensors[[0, 4, 9]] = 1.0  # constant sensors, like in the real data
        rows.append([eid, c, *settings, *sensors])
    return rows


def _write(path: Path, rows) -> None:
    path.write_text("\n".join(" ".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in r) + " "
                              for r in rows) + "\n")


@pytest.fixture(scope="session")
def cmapss_dir(tmp_path_factory) -> Path:
    """FD001-style files: 10 train engines run to failure, 4 truncated test engines (one short)."""
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("cmapss")
    train = []
    for eid in range(1, 11):
        life = int(rng.integers(45, 70))
        train += _engine_rows(eid, life, life, rng)
    test, rul = [], []
    for eid, (life, seen) in enumerate([(60, 40), (55, 12), (70, 66), (50, 30)], start=1):
        test += _engine_rows(eid, seen, life, rng)
        rul.append(life - seen)
    _write(d / "train_FD001.txt", [[int(r[0]), int(r[1]), *map(float, r[2:])] for r in train])
    _write(d / "test_FD001.txt", [[int(r[0]), int(r[1]), *map(float, r[2:])] for r in test])
    (d / "RUL_FD001.txt").write_text("\n".join(str(r) for r in rul) + "\n")
    return d
