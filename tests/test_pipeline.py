"""End-to-end: train.py on synthetic data, predict.py must reproduce it, baselines.py must run on it."""

import json

import pandas as pd

import baselines
import predict
import train

ARTIFACTS = ["best_model.pt", "training_config.json", "scaler.joblib", "train_history.csv",
             "test_predictions.csv", "metrics.json", "training_loss.png", "validation_loss.png",
             "optuna_trials.csv", "best_params.json", "test_pred_vs_true.png", "rul_trajectories.png"]


def test_train_then_predict(cmapss_dir, tmp_path):
    out = tmp_path / "run"
    metrics = train.main(["--dataset_dir", str(cmapss_dir), "--dataset", "FD001", "--seq_len", "20",
                          "--epochs", "2", "--trials", "2", "--search", "basic", "--ensemble", "2",
                          "--d_model", "16", "--device", "cpu", "--output_dir", str(out)])
    for name in ARTIFACTS:
        assert (out / name).is_file(), name
    assert metrics["test"]["n"] == 4
    preds = pd.read_csv(out / "test_predictions.csv")
    assert preds["engine_id"].tolist() == [1, 2, 3, 4]
    assert preds["padded"].tolist() == [False, True, False, False]  # engine 2 has 12 < 20 cycles
    assert json.loads((out / "training_config.json").read_text())["dataset"] == "FD001"

    csv = tmp_path / "pred.csv"
    predict.main(["--model", str(out / "best_model.pt"), "--scaler", str(out / "scaler.joblib"),
                  "--test_file", str(cmapss_dir / "test_FD001.txt"),
                  "--rul_file", str(cmapss_dir / "RUL_FD001.txt"), "--output", str(csv), "--device", "cpu"])
    standalone = pd.read_csv(csv)
    assert standalone["engine_id"].tolist() == preds["engine_id"].tolist()
    assert (standalone["predicted_rul"] - preds["predicted_rul"]).abs().max() < 1e-3

    res = baselines.main(["--transformer_dir", str(out), "--dataset_dir", str(cmapss_dir), "--lstm_trials", "1",
                          "--ensemble", "1", "--epochs", "2", "--device", "cpu"])
    assert len(res) == 5 and res["test_rmse"].notna().all()
    assert {"Mean predictor", "Linear regression (ridge)", "Gradient boosting (XGBoost)"} <= set(res["model"])
    for name in ["baseline_results.csv", "baseline_test_predictions.csv", "baseline_comparison.png"]:
        assert (out / "baselines" / name).is_file(), name
    # the Transformer row must be copied from the run, not recomputed differently
    tf = res[res["model"].str.startswith("Transformer")].iloc[0]
    assert tf["test_rmse"] == metrics["test"]["rmse"]
