import numpy as np
import pandas as pd
import pytest

from rul import data as D


def test_load_assigns_26_columns_and_maps_rul(cmapss_dir):
    train, test, true_rul = D.load_dataset(cmapss_dir, "FD001")
    assert list(train.columns) == D.CMAPSS_COLUMNS and len(D.CMAPSS_COLUMNS) == 26
    assert train["engine_id"].nunique() == 10
    assert list(true_rul.index) == [1, 2, 3, 4]
    assert true_rul.loc[2] == 55 - 12  # row i of RUL file belongs to engine i


def test_missing_dataset_raises_clear_error(tmp_path):
    with pytest.raises(D.DatasetNotFoundError, match="download_data.py"):
        D.check_dataset(tmp_path, "FD003")


def test_train_rul_and_clip():
    df = pd.DataFrame({"engine_id": [1] * 5 + [2] * 3, "cycle": [1, 2, 3, 4, 5, 1, 2, 3]})
    out = D.add_train_rul(df, rul_clip=None)
    assert out["RUL"].tolist() == [4, 3, 2, 1, 0, 2, 1, 0]
    assert D.add_train_rul(df, rul_clip=2)["RUL"].max() == 2


def test_engine_split_has_no_leakage_and_is_deterministic():
    ids = range(1, 101)
    tr, va = D.split_engines(ids, 0.2, seed=7)
    assert len(va) == 20 and len(tr) == 80
    assert not set(tr) & set(va) and set(tr) | set(va) == set(ids)
    assert (tr, va) == D.split_engines(ids, 0.2, seed=7)


def test_constant_columns_and_settings_are_dropped(cmapss_dir):
    train, _, _ = D.load_dataset(cmapss_dir, "FD001")
    feats = D.select_features(train)
    assert not {"sensor_1", "sensor_5", "sensor_10"} & set(feats)
    assert not any(f.startswith("op_setting") for f in feats)
    assert "op_setting_1" in D.select_features(train, use_settings=True)


def _toy(n1=6, n2=4):
    df = pd.DataFrame({"engine_id": [1] * n1 + [2] * n2,
                       "cycle": list(range(1, n1 + 1)) + list(range(1, n2 + 1))})
    df["f"] = np.arange(len(df), dtype=np.float32)  # unique value per row
    return D.add_train_rul(df, rul_clip=None)


def test_full_windows_stay_within_engine_and_label_last_cycle():
    df = _toy()
    X, y, ids = D.make_train_windows(df, ["f"], seq_len=3, include_partial=False)
    assert X.shape == (4 + 2, 3, 1)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
    assert y[0] == 3  # RUL at cycle 3 of a 6-cycle engine
    assert ids.tolist() == [1, 1, 1, 1, 2, 2]
    np.testing.assert_array_equal(X[4, :, 0], [6, 7, 8])  # first window of engine 2 starts at its own row
    assert y[-1] == 0


def test_partial_windows_end_at_every_cycle_and_are_padded_like_test_data():
    df = _toy(n1=6, n2=2)
    X, y, ids = D.make_train_windows(df, ["f"], seq_len=4)
    assert (ids == 1).sum() == 6 and (ids == 2).sum() == 2  # one window per cycle
    np.testing.assert_array_equal(X[0, :, 0], [0, 0, 0, 0])  # ends at cycle 1
    assert y[0] == 5
    eng2 = X[ids == 2]
    np.testing.assert_array_equal(eng2[-1, :, 0], [6, 6, 6, 7])
    # the window ending at an engine's last cycle equals what inference builds for it
    X_last, _, _, _ = D.make_last_windows(df, ["f"], seq_len=4)
    np.testing.assert_array_equal(eng2[-1], X_last[1])
    assert y[ids == 2].tolist() == [1, 0]


def test_last_windows_one_per_engine_ending_at_final_cycle():
    df = _toy(n1=6, n2=2)
    X, ids, last_cycle, n = D.make_last_windows(df, ["f"], seq_len=3)
    assert ids.tolist() == [1, 2] and last_cycle.tolist() == [6, 2] and n.tolist() == [6, 2]
    np.testing.assert_array_equal(X[0, :, 0], [3, 4, 5])
    np.testing.assert_array_equal(X[1, :, 0], [6, 6, 7])


def test_test_rul_trajectory_matches_file(cmapss_dir):
    _, test, true_rul = D.load_dataset(cmapss_dir, "FD001")
    traj = D.add_test_rul(test, true_rul)
    last = traj.groupby("engine_id").tail(1).set_index("engine_id")["RUL"]
    np.testing.assert_array_equal(last.loc[true_rul.index].to_numpy(), true_rul.to_numpy())
