"""Standalone engine-level RUL inference with a trained checkpoint.

Example:
    python predict.py --model outputs/FD001/best_model.pt --scaler outputs/FD001/scaler.joblib \
        --test_file data/CMAPSS/test_FD001.txt --rul_file data/CMAPSS/RUL_FD001.txt \
        --output predictions.csv

Builds the same scaled, fixed-length sequences as training (via rul.data) and
outputs one prediction per engine using the window ending at its last cycle.
"""
from __future__ import annotations

import argparse
import json
import sys

import joblib
import pandas as pd
import torch

from rul import data as D
from rul.engine import predict, regression_metrics
from rul.model import load_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description="Predict RUL per engine from a CMAPSS-format file.")
    p.add_argument("--model", required=True, help="path to best_model.pt")
    p.add_argument("--scaler", required=True, help="path to scaler.joblib")
    p.add_argument("--test_file", required=True, help="CMAPSS-format txt (26 whitespace-separated columns)")
    p.add_argument("--seq_len", type=int, default=None, help="must match training (default: from checkpoint)")
    p.add_argument("--rul_file", default=None, help="optional RUL_FDxxx.txt to report metrics")
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    if a.device.startswith("cuda"):  # same matmul precision as train.py, so GPU results match exactly
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, ckpt = load_checkpoint(a.model, map_location=a.device)
    model.to(a.device)
    seq_len = a.seq_len or ckpt["seq_len"]
    if seq_len != ckpt["seq_len"]:
        sys.exit(f"ERROR: --seq_len {seq_len} differs from the training sequence length {ckpt['seq_len']}")
    scaler = joblib.load(a.scaler)
    features = ckpt["features"]
    if list(getattr(scaler, "feature_names_in_", features)) != features:
        sys.exit("ERROR: scaler features do not match the checkpoint features")

    df = D.scale(D.load_cmapss_file(a.test_file), scaler, features)
    X, eng_ids, last_cycle, n_cycles = D.make_last_windows(df, features, seq_len)
    y_pred = predict(model, X, a.device, ckpt["target_scale"], clip_max=ckpt["rul_clip"])
    out = pd.DataFrame({"engine_id": eng_ids, "last_cycle": last_cycle, "n_cycles": n_cycles,
                        "predicted_rul": y_pred.round(3)})

    if a.rul_file:
        true = D.load_rul_file(a.rul_file)
        if len(true) != len(eng_ids):
            sys.exit(f"ERROR: RUL file has {len(true)} rows, test file has {len(eng_ids)} engines")
        out["true_rul"] = true
        out["error"] = (out["predicted_rul"] - out["true_rul"]).round(3)
        print(json.dumps(regression_metrics(true, y_pred), indent=2))

    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out)} engine predictions to {a.output}")


if __name__ == "__main__":
    main()
