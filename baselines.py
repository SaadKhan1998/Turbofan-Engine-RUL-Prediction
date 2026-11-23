"""Compare the trained Transformer with simpler baseline models on identical data.

Every baseline uses exactly the same engines, train/validation split, scaling,
window length and test windows as the Transformer run in ``--transformer_dir``
(read from its ``training_config.json``). Baseline hyper-parameters are chosen on
the validation engines only; the test set is scored once at the end.

Baselines:
    mean      - always predicts the average training RUL (no learning)
    ridge     - linear regression with L2 penalty on the flattened window
    boosting  - XGBoost gradient-boosted trees on the flattened window
    lstm      - recurrent network, tuned with Optuna and ensembled like the Transformer

Example:
    python baselines.py --transformer_dir outputs/FD001 --dataset_dir data/CMAPSS
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from rul import data as D
from rul import plots
from rul.engine import predict, regression_metrics, set_seed, train_model
from rul.model import LSTMRegressor, RULEnsemble


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Baseline comparison for a trained Transformer run.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--transformer_dir", required=True, help="output folder of a train.py run")
    p.add_argument("--dataset_dir", default=None, help="default: the dataset_dir recorded in the run")
    p.add_argument("--output_dir", default=None, help="default: <transformer_dir>/baselines")
    p.add_argument("--models", nargs="+", default=["mean", "ridge", "boosting", "lstm"],
                   choices=["mean", "ridge", "boosting", "lstm"])
    p.add_argument("--lstm_trials", type=int, default=15, help="Optuna trials for the LSTM")
    p.add_argument("--ensemble", type=int, default=None, help="LSTM ensemble size (default: same as the run)")
    p.add_argument("--epochs", type=int, default=None, help="max LSTM epochs (default: same as the run)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def flat(X: np.ndarray) -> np.ndarray:
    """[N, seq_len, F] -> [N, seq_len * F]: the input format of non-sequence models."""
    return X.reshape(len(X), -1)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    run = Path(args.transformer_dir)
    cfg = json.loads((run / "training_config.json").read_text())
    tf_metrics = json.loads((run / "metrics.json").read_text())
    out = Path(args.output_dir or run / "baselines")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed, rul_clip = cfg["seed"], cfg["rul_clip"] if cfg["rul_clip"] > 0 else None
    seq_len = int(cfg["best_hyperparameters"]["seq_len"])
    target_scale = cfg["target_scale"]
    ensemble = args.ensemble or cfg["ensemble"]
    epochs = args.epochs or cfg["epochs"]
    P = D.prepare_data(args.dataset_dir or cfg["dataset_dir"], cfg["dataset"], rul_clip, cfg["val_fraction"],
                       seed, cfg["use_settings"])
    assert P.features == cfg["features"] and P.val_ids == cfg["val_engines"], "data differs from the run"
    w = P.windows(seq_len)
    X_te, eng_ids, _, _, y_true = P.test_windows(seq_len)
    capped = np.minimum(y_true, rul_clip) if rul_clip else y_true
    clip = lambda p: np.clip(p, 0, rul_clip)
    print(f"{cfg['dataset']}: seq_len {seq_len}, {len(P.features)} features, "
          f"{len(w['X_train'])} train / {len(w['X_val'])} val windows, {len(X_te)} test engines")

    rows, test_preds = [], {"engine_id": eng_ids, "true_rul": y_true}

    def record(name: str, val_pred, test_pred, fit_seconds: float, params: int | None, notes: str):
        test_preds[name] = np.round(test_pred, 3)
        t, c = regression_metrics(y_true, test_pred), regression_metrics(capped, test_pred)
        rows.append(dict(model=name, val_rmse=regression_metrics(w["y_val"], val_pred)["rmse"],
                         test_rmse=t["rmse"], test_mae=t["mae"], nasa_score=t["nasa_score"],
                         test_rmse_capped=c["rmse"], fit_seconds=round(fit_seconds, 1),
                         parameters=params, notes=notes))
        print(f"  {name:<28} val RMSE {rows[-1]['val_rmse']:6.2f} | test RMSE {t['rmse']:6.2f} "
              f"| MAE {t['mae']:6.2f} | NASA {t['nasa_score']:9.1f} | {fit_seconds:6.1f}s", flush=True)

    # ------------------------------------------------------------------ mean
    if "mean" in args.models:
        m = float(w["y_train"].mean())
        record("Mean predictor", np.full(len(w["y_val"]), m), np.full(len(y_true), m), 0.0, 1,
               f"always predicts {m:.1f} cycles")

    Xtr_f, Xva_f, Xte_f = flat(w["X_train"]), flat(w["X_val"]), flat(X_te)

    # ----------------------------------------------------------------- ridge
    if "ridge" in args.models:
        t0, best = time.time(), None
        for alpha in [0.01, 0.1, 1, 10, 100, 1000]:
            r = Ridge(alpha=alpha).fit(Xtr_f, w["y_train"])
            v = regression_metrics(w["y_val"], clip(r.predict(Xva_f)))["rmse"]
            if best is None or v < best[0]:
                best = (v, alpha, r)
        _, alpha, r = best
        record("Linear regression (ridge)", clip(r.predict(Xva_f)), clip(r.predict(Xte_f)), time.time() - t0,
               r.coef_.size + 1, f"alpha={alpha} chosen on validation from 6 values")

    # -------------------------------------------------------------- boosting
    if "boosting" in args.models:
        from xgboost import XGBRegressor

        t0, best = time.time(), None
        xgb_device = "cuda" if device.type == "cuda" else "cpu"
        for lr in [0.05, 0.1]:
            for depth in [4, 6]:
                g = XGBRegressor(n_estimators=300, learning_rate=lr, max_depth=depth, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", device=xgb_device, random_state=seed)
                g.fit(Xtr_f, w["y_train"])
                g.set_params(device="cpu")  # predict from numpy on CPU (avoids device-mismatch copies)
                v = regression_metrics(w["y_val"], clip(g.predict(Xva_f)))["rmse"]
                if best is None or v < best[0]:
                    best = (v, lr, depth, g)
        _, lr, depth, g = best
        record("Gradient boosting (XGBoost)", clip(g.predict(Xva_f)), clip(g.predict(Xte_f)), time.time() - t0,
               None, f"learning_rate={lr}, max_depth={depth}, 300 trees, chosen on validation from 4 configs")

    # ------------------------------------------------------------------ lstm
    if "lstm" in args.models:
        import optuna

        t0 = time.time()

        def fit(hp, s, trial=None, verbose=False):
            kw = dict(n_features=len(P.features), hidden_size=hp["hidden_size"], num_layers=hp["num_layers"],
                      dropout=hp["dropout"])
            return train_model(w, kw, lr=hp["lr"], batch_size=hp["batch_size"], epochs=epochs,
                               patience=cfg["patience"], weight_decay=hp["weight_decay"], loss=cfg["loss"],
                               device=device, seed=s, target_scale=target_scale, trial=trial, verbose=verbose,
                               model_cls=LSTMRegressor)

        def objective(trial):
            hp = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                      batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
                      hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
                      num_layers=trial.suggest_categorical("num_layers", [1, 2]),
                      dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
                      weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True))
            _, hist, best_ep, best_val = fit(hp, seed, trial)
            print(f"  [lstm trial {trial.number:2d}] {trial.params} -> val RMSE {hist[best_ep - 1]['val_rmse']:.2f}",
                  flush=True)
            return best_val

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8))
        study.optimize(objective, n_trials=args.lstm_trials)
        hp = study.best_params
        members = [fit(hp, seed + i)[0] for i in range(ensemble)]
        ens = members[0] if ensemble == 1 else RULEnsemble(members)
        single = [regression_metrics(y_true, predict(m, X_te, device, target_scale, clip_max=rul_clip))["rmse"]
                  for m in members]
        n_params = sum(p.numel() for p in members[0].parameters())
        record(f"LSTM (ensemble of {ensemble})", predict(ens, w["X_val"], device, target_scale, clip_max=rul_clip),
               predict(ens, X_te, device, target_scale, clip_max=rul_clip), time.time() - t0, n_params * ensemble,
               f"{args.lstm_trials} Optuna trials; best {hp}; single members test RMSE "
               f"{min(single):.2f}-{max(single):.2f}")

    # ----------------------------------------------------------- transformer
    tf_single = [m["rmse"] for m in tf_metrics["test_single_members"]]
    tf_pred = pd.read_csv(run / "test_predictions.csv").set_index("engine_id").loc[eng_ids, "predicted_rul"]
    test_preds["Transformer"] = tf_pred.to_numpy()
    rows.append(dict(model=f"Transformer (ensemble of {len(tf_single)}, this repo)",
                     val_rmse=tf_metrics["validation_all_windows"]["rmse"], test_rmse=tf_metrics["test"]["rmse"],
                     test_mae=tf_metrics["test"]["mae"], nasa_score=tf_metrics["test"]["nasa_score"],
                     test_rmse_capped=tf_metrics["test_vs_capped_truth"]["rmse"],
                     fit_seconds=tf_metrics["runtime_seconds"], parameters=None,
                     notes=f"Optuna search; single members test RMSE {min(tf_single):.2f}-{max(tf_single):.2f}"))

    res = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    res.to_csv(out / "baseline_results.csv", index=False)
    pd.DataFrame(test_preds).to_csv(out / "baseline_test_predictions.csv", index=False)
    (out / "baseline_results.json").write_text(json.dumps(
        {"dataset": cfg["dataset"], "seq_len": seq_len, "results": res.to_dict(orient="records")}, indent=2))
    plots.model_comparison(res, out / "baseline_comparison.png",
                           f"{cfg['dataset']} test RMSE by model (lower is better)", highlight="Transformer")
    print("\n" + res[["model", "val_rmse", "test_rmse", "test_mae", "nasa_score", "test_rmse_capped"]]
          .round(2).to_string(index=False))
    print(f"Written to {out}")
    return res


if __name__ == "__main__":
    set_seed(0)
    main()
