"""CMAPSS loading, labelling, scaling and windowing.

Every function here is shared by ``train.py`` and ``predict.py`` so that
inference builds exactly the same sequences the model was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ID_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS  # 26 columns
CANDIDATE_FEATURES = SETTING_COLUMNS + SENSOR_COLUMNS

SUPPORTED_DATASETS = ("FD001", "FD003")


class DatasetNotFoundError(FileNotFoundError):
    pass


def dataset_files(dataset_dir: str | Path, dataset: str) -> dict[str, Path]:
    d = Path(dataset_dir)
    return {
        "train": d / f"train_{dataset}.txt",
        "test": d / f"test_{dataset}.txt",
        "rul": d / f"RUL_{dataset}.txt",
    }


def check_dataset(dataset_dir: str | Path, dataset: str) -> dict[str, Path]:
    files = dataset_files(dataset_dir, dataset)
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise DatasetNotFoundError(
            "CMAPSS dataset files not found:\n  "
            + "\n  ".join(missing)
            + "\nDownload them with:\n"
            f"  python download_data.py --dataset_dir {dataset_dir}\n"
            "or pass --download to train.py, or point --dataset_dir at a folder "
            f"containing train_{dataset}.txt, test_{dataset}.txt and RUL_{dataset}.txt."
        )
    return files


def load_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read a whitespace-separated CMAPSS train/test file and assign the 26 columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")  # trailing spaces produce empty columns
    if df.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(f"{path}: expected {len(CMAPSS_COLUMNS)} columns, found {df.shape[1]}")
    df.columns = CMAPSS_COLUMNS
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def load_rul_file(path: str | Path) -> np.ndarray:
    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    return rul.iloc[:, 0].to_numpy(dtype=np.float32)


def load_dataset(dataset_dir: str | Path, dataset: str):
    files = check_dataset(dataset_dir, dataset)
    train = load_cmapss_file(files["train"])
    test = load_cmapss_file(files["test"])
    rul = load_rul_file(files["rul"])
    test_engines = np.sort(test["engine_id"].unique())
    if len(rul) != len(test_engines):
        raise ValueError(f"RUL file has {len(rul)} rows but test file has {len(test_engines)} engines")
    # Row i of RUL_FDxxx.txt is the true RUL of the i-th test engine (sorted by id).
    true_rul = pd.Series(rul, index=test_engines, name="true_rul")
    return train, test, true_rul


def add_train_rul(df: pd.DataFrame, rul_clip: float | None) -> pd.DataFrame:
    """RUL at each cycle = last observed cycle of that engine - current cycle."""
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).astype(np.float32)
    if rul_clip is not None:
        df["RUL"] = df["RUL"].clip(upper=rul_clip)
    return df


def add_test_rul(test_df: pd.DataFrame, true_rul: pd.Series) -> pd.DataFrame:
    """True RUL at every observed test cycle, derived from the end-of-record RUL file."""
    df = test_df.copy()
    last = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = (df["engine_id"].map(true_rul) + last - df["cycle"]).astype(np.float32)
    return df


def select_features(train_df: pd.DataFrame, use_settings: bool = False, tol: float = 1e-6) -> list[str]:
    """Pick input columns: sensors (optionally operating settings) that vary in the training data.

    FD001/FD003 run at a single operating condition, so the settings are only
    small noise around a constant and are excluded by default.
    """
    candidates = CANDIDATE_FEATURES if use_settings else SENSOR_COLUMNS
    std = train_df[candidates].std()
    return [c for c in candidates if std[c] > tol]


def split_engines(engine_ids, val_fraction: float, seed: int):
    """Split by whole engines so no engine appears in both train and validation."""
    ids = np.array(sorted(set(int(e) for e in engine_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    return sorted(ids[n_val:].tolist()), sorted(ids[:n_val].tolist())


def scale(df: pd.DataFrame, scaler, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[features] = scaler.transform(df[features]).astype(np.float32)
    return df


@dataclass
class PreparedData:
    """Everything downstream code needs, built identically for every model."""

    train_raw: pd.DataFrame  # unscaled training data with RUL labels
    test_raw: pd.DataFrame
    tr_df: pd.DataFrame  # scaled training engines
    va_df: pd.DataFrame  # scaled validation engines
    test_df: pd.DataFrame  # scaled test engines
    true_rul: pd.Series  # true test RUL indexed by engine id
    features: list[str]
    scaler: MinMaxScaler
    train_ids: list[int]
    val_ids: list[int]
    _cache: dict = field(default_factory=dict, repr=False)

    def windows(self, seq_len: int) -> dict:
        """Train/validation windows for ``seq_len`` (cached)."""
        if seq_len not in self._cache:
            X_tr, y_tr, _ = make_train_windows(self.tr_df, self.features, seq_len)
            X_va, y_va, _ = make_train_windows(self.va_df, self.features, seq_len)
            self._cache[seq_len] = dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va)
        return self._cache[seq_len]

    def test_windows(self, seq_len: int):
        """One window per test engine ending at its last cycle, plus the matching true RUL."""
        X, ids, last_cycle, n_cycles = make_last_windows(self.test_df, self.features, seq_len)
        return X, ids, last_cycle, n_cycles, self.true_rul.loc[ids].to_numpy()


def prepare_data(dataset_dir, dataset: str, rul_clip: float | None, val_fraction: float, seed: int,
                 use_settings: bool = False) -> PreparedData:
    """Load, label, split by engine, select features and scale (scaler fitted on training engines only)."""
    train_raw, test_raw, true_rul = load_dataset(dataset_dir, dataset)
    train_raw = add_train_rul(train_raw, rul_clip)
    train_ids, val_ids = split_engines(train_raw["engine_id"].unique(), val_fraction, seed)
    assert not set(train_ids) & set(val_ids), "engine leakage between train and validation"
    tr_df = train_raw[train_raw["engine_id"].isin(train_ids)]
    va_df = train_raw[train_raw["engine_id"].isin(val_ids)]
    features = select_features(tr_df, use_settings=use_settings)
    scaler = MinMaxScaler().fit(tr_df[features])
    return PreparedData(train_raw=train_raw, test_raw=test_raw, tr_df=scale(tr_df, scaler, features),
                        va_df=scale(va_df, scaler, features), test_df=scale(test_raw, scaler, features),
                        true_rul=true_rul, features=features, scaler=scaler, train_ids=train_ids, val_ids=val_ids)


def _pad_front(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Left-pad a short engine history by repeating its first observed row."""
    if len(values) >= seq_len:
        return values
    pad = np.repeat(values[:1], seq_len - len(values), axis=0)
    return np.concatenate([pad, values], axis=0)


def make_train_windows(df: pd.DataFrame, features: list[str], seq_len: int, include_partial: bool = True):
    """Sliding windows per engine; label = RUL at the window's last cycle.

    With ``include_partial`` (default) there is one window ending at *every* cycle:
    windows ending before cycle ``seq_len`` are front-padded by repeating the first
    row, exactly as short test engines are padded at inference time. Otherwise only
    full windows are made (engines shorter than ``seq_len`` still yield one padded window).

    Returns X [N, seq_len, F], y [N], engine id per window [N].
    """
    xs, ys, ids = [], [], []
    for eid, g in df.groupby("engine_id", sort=True):
        raw = g[features].to_numpy(np.float32)
        vals = _pad_front(raw, len(raw) + seq_len - 1) if include_partial else _pad_front(raw, seq_len)
        rul = g["RUL"].to_numpy(np.float32)
        rul = np.concatenate([np.full(len(vals) - len(rul), rul[0], np.float32), rul])
        win = np.lib.stride_tricks.sliding_window_view(vals, seq_len, axis=0)  # [n, F, L]
        xs.append(win.transpose(0, 2, 1))
        ys.append(rul[seq_len - 1:])
        ids.append(np.full(len(win), eid, dtype=np.int64))
    return (np.ascontiguousarray(np.concatenate(xs)).astype(np.float32),
            np.concatenate(ys).astype(np.float32),
            np.concatenate(ids))


def make_last_windows(df: pd.DataFrame, features: list[str], seq_len: int):
    """One window per engine ending at its final available cycle.

    Returns X [n_engines, seq_len, F], engine ids [n_engines] (sorted),
    last cycle per engine, and the original number of observed cycles.
    """
    xs, ids, last_cycle, n_cycles = [], [], [], []
    for eid, g in df.groupby("engine_id", sort=True):
        vals = g[features].to_numpy(np.float32)
        xs.append(_pad_front(vals, seq_len)[-seq_len:])
        ids.append(int(eid))
        last_cycle.append(int(g["cycle"].iloc[-1]))
        n_cycles.append(len(g))
    return np.stack(xs).astype(np.float32), np.array(ids), np.array(last_cycle), np.array(n_cycles)
