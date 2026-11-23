"""End-to-end training of a Transformer RUL model on NASA CMAPSS FD001/FD003.

Pipeline: load data -> engine-level train/val split -> scale (train engines only)
-> windows -> Optuna search -> final seed ensemble -> test evaluation -> artifacts.

Example:
    python train.py --dataset_dir data/CMAPSS --dataset FD001 --epochs 50 --trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from rul import data as D
from rul import plots
from rul.engine import predict, regression_metrics, set_seed, train_model
from rul.model import RULEnsemble, save_checkpoint


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer RUL model on NASA CMAPSS.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", choices=D.SUPPORTED_DATASETS, default="FD001")
    g.add_argument("--dataset_dir", default="data/CMAPSS", help="folder with train_/test_/RUL_FDxxx.txt")
    g.add_argument("--download", action="store_true", help="download CMAPSS into --dataset_dir if missing")
    g.add_argument("--seq_len", type=int, default=30, help="window length in cycles")
    g.add_argument("--rul_clip", type=float, default=125.0, help="piecewise-linear RUL cap; <=0 disables")
    g.add_argument("--val_fraction", type=float, default=0.2, help="fraction of training ENGINES held out")
    g.add_argument("--use_settings", action="store_true", help="also feed the 3 operating settings")

    g = p.add_argument_group("training")
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=1e-3)
    g.add_argument("--epochs", type=int, default=50, help="max epochs per model")
    g.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    g.add_argument("--loss", choices=["mse", "huber"], default="mse")
    g.add_argument("--weight_decay", type=float, default=1e-4)
    g.add_argument("--ensemble", type=int, default=5, help="seeds trained for the final model (averaged)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = p.add_argument_group("model")
    g.add_argument("--d_model", type=int, default=64)
    g.add_argument("--nhead", type=int, default=4)
    g.add_argument("--num_layers", type=int, default=2, help="encoder layers (feed-forward width = 2 x d_model)")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--pooling", choices=["mean", "last"], default="mean")

    g = p.add_argument_group("hyper-parameter search")
    g.add_argument("--trials", type=int, default=10, help="Optuna trials (0 = use the values above)")
    g.add_argument("--search", choices=["basic", "full"], default="full",
                   help="basic: lr + batch size; full: also seq_len, d_model, layers, dropout, weight decay")
    g.add_argument("--trial_epochs", type=int, default=None, help="max epochs per trial (default: --epochs)")

    p.add_argument("--output_dir", default=None, help="default: outputs/<dataset>_<timestamp>")
    a = p.parse_args(argv)
    if min(a.seq_len, a.epochs, a.batch_size, a.ensemble) < 1 or a.trials < 0 or a.lr <= 0:
        p.error("seq_len, epochs, batch_size, ensemble must be >= 1; trials >= 0; lr > 0")
    if a.d_model % a.nhead:
        p.error("--d_model must be divisible by --nhead")
    return a


SEARCH = {
    "basic": {"batch_size": [32, 64, 128, 256]},
    "full": {"batch_size": [64, 128, 256], "seq_len": [30, 40, 50], "d_model": [32, 64, 128],
             "num_layers": [1, 2, 3]},
}


def suggest(trial, mode: str) -> dict:
    s = SEARCH[mode]
    hp = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
          "batch_size": trial.suggest_categorical("batch_size", s["batch_size"])}
    if mode == "full":
        hp.update(seq_len=trial.suggest_categorical("seq_len", s["seq_len"]),
                  d_model=trial.suggest_categorical("d_model", s["d_model"]),
                  num_layers=trial.suggest_categorical("num_layers", s["num_layers"]),
                  dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
                  weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True))
    return hp


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rul_clip = args.rul_clip if args.rul_clip > 0 else None

    # ------------------------------------------------------------------ data
    try:
        D.check_dataset(args.dataset_dir, args.dataset)
    except D.DatasetNotFoundError:
        if not args.download:
            raise
        from download_data import download

        download(args.dataset_dir)

    out = Path(args.output_dir or f"outputs/{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({dev_name}) | output: {out}")

    P = D.prepare_data(args.dataset_dir, args.dataset, rul_clip, args.val_fraction, args.seed, args.use_settings)
    train_raw, test_raw, true_rul, features, scaler = P.train_raw, P.test_raw, P.true_rul, P.features, P.scaler
    va_df, test_df, train_ids, val_ids = P.va_df, P.test_df, P.train_ids, P.val_ids
    windows = P.windows
    print(f"{args.dataset}: {len(train_ids)} train / {len(val_ids)} val / {test_raw['engine_id'].nunique()} "
          f"test engines | {len(features)} features: {', '.join(features)}")

    target_scale = rul_clip or float(train_raw["RUL"].max())
    base_hp = dict(lr=args.lr, batch_size=args.batch_size, seq_len=args.seq_len, d_model=args.d_model,
                   num_layers=args.num_layers, dropout=args.dropout, weight_decay=args.weight_decay)

    def fit(hp: dict, seed: int, epochs: int, trial=None, verbose=False):
        model_kwargs = dict(n_features=len(features), d_model=hp["d_model"], nhead=args.nhead,
                            num_layers=hp["num_layers"], dim_feedforward=2 * hp["d_model"],
                            dropout=hp["dropout"], pooling=args.pooling, max_len=max(1024, hp["seq_len"]))
        return train_model(data=windows(hp["seq_len"]), model_kwargs=model_kwargs, lr=hp["lr"],
                           batch_size=hp["batch_size"], epochs=epochs, patience=args.patience,
                           weight_decay=hp["weight_decay"], loss=args.loss, device=device, seed=seed,
                           target_scale=target_scale, trial=trial, verbose=verbose)

    # ---------------------------------------------------------------- Optuna
    best_hp, trials_info, t_start = dict(base_hp), [], time.time()
    if args.trials > 0:
        import optuna

        trial_epochs = args.trial_epochs or args.epochs

        def objective(trial):
            hp = {**base_hp, **suggest(trial, args.search)}
            # Re-use an identical finished trial of this study instead of retraining (no global cache).
            for t in trial.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
                if t.params == trial.params:
                    print(f"[trial {trial.number}] duplicate of trial {t.number}, reusing its value")
                    return t.value
            t0 = time.time()
            _, hist, best_ep, best_val = fit(hp, args.seed, trial_epochs, trial=trial)
            h = hist[best_ep - 1]
            trial.set_user_attr("best_epoch", best_ep)
            trial.set_user_attr("val_rmse", h["val_rmse"])
            trial.set_user_attr("val_mae", h["val_mae"])
            print(f"[trial {trial.number:2d}] {trial.params} -> val_loss {best_val:.5f} "
                  f"(RMSE {h['val_rmse']:.2f}) epoch {best_ep}/{len(hist)}, {time.time() - t0:.0f}s", flush=True)
            return best_val

        def log_pruned(study, trial):
            if trial.state == optuna.trial.TrialState.PRUNED:
                print(f"[trial {trial.number:2d}] {trial.params} -> pruned", flush=True)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8))
        first = {k: base_hp[k] for k in ("lr", "batch_size")}
        if args.search == "full":
            first.update({k: base_hp[k] for k in ("seq_len", "d_model", "num_layers", "dropout", "weight_decay")})
        if all(first[k] in v for k, v in SEARCH[args.search].items()):
            study.enqueue_trial(first, skip_if_exists=True)
        study.optimize(objective, n_trials=args.trials, callbacks=[log_pruned])
        best_hp.update(study.best_params)
        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs", "duration"))
        trials_df["duration"] = trials_df["duration"].dt.total_seconds().round(1)
        trials_df.to_csv(out / "optuna_trials.csv", index=False)
        trials_info = trials_df.to_dict(orient="records")
        print(f"Optuna finished in {time.time() - t_start:.0f}s. Best trial {study.best_trial.number}: "
              f"{study.best_params} (val_loss {study.best_value:.5f})")
        (out / "best_params.json").write_text(json.dumps(
            {"best_params": study.best_params, "best_val_loss": study.best_value,
             "best_trial": study.best_trial.number}, indent=2))

    # ------------------------------------------------- final seed ensemble
    print(f"Training final ensemble of {args.ensemble} with {best_hp}")
    members, histories, member_info = [], [], []
    for i in range(args.ensemble):
        seed = args.seed + i
        print(f"-- member {i + 1}/{args.ensemble} (seed {seed})")
        model, hist, best_ep, best_val = fit(best_hp, seed, args.epochs, verbose=True)
        members.append(model)
        histories.append(pd.DataFrame(hist).assign(member=i + 1, seed=seed))
        member_info.append(dict(member=i + 1, seed=seed, best_epoch=best_ep, epochs_run=len(hist),
                                best_val_loss=best_val, val_rmse=hist[best_ep - 1]["val_rmse"]))
    ensemble = members[0] if len(members) == 1 else RULEnsemble(members)
    seq_len = int(best_hp["seq_len"])

    hist_df = pd.concat(histories, ignore_index=True)
    hist_df.to_csv(out / "train_history.csv", index=False)
    plots.loss_curve(hist_df, "train_loss", "Training loss", out / "training_loss.png")
    plots.loss_curve(hist_df, "val_loss", "Validation loss", out / "validation_loss.png")

    save_checkpoint(out / "best_model.pt", members, features=features, seq_len=seq_len, rul_clip=rul_clip,
                    target_scale=target_scale, dataset=args.dataset, hyperparameters=best_hp, members=member_info)
    joblib.dump(scaler, out / "scaler.joblib")

    # -------------------------------------- validation + test evaluation
    w = windows(seq_len)
    val_pred = predict(ensemble, w["X_val"], device, target_scale, clip_max=rul_clip)
    X_va_last, va_last_ids, _, _ = D.make_last_windows(va_df, features, seq_len)
    va_last_true = va_df.groupby("engine_id")["RUL"].last().loc[va_last_ids].to_numpy()

    X_te, eng_ids, last_cycle, n_cycles = D.make_last_windows(test_df, features, seq_len)
    y_pred = predict(ensemble, X_te, device, target_scale, clip_max=rul_clip)
    y_true = true_rul.loc[eng_ids].to_numpy()
    member_preds = np.stack([predict(m, X_te, device, target_scale, clip_max=rul_clip) for m in members])
    pd.DataFrame({"engine_id": eng_ids, "last_cycle": last_cycle, "n_cycles": n_cycles,
                  "padded": n_cycles < seq_len, "true_rul": y_true, "predicted_rul": np.round(y_pred, 3),
                  "error": np.round(y_pred - y_true, 3),
                  "member_std": np.round(member_preds.std(axis=0), 3)}).to_csv(out / "test_predictions.csv",
                                                                               index=False)

    capped_truth = np.minimum(y_true, rul_clip) if rul_clip else y_true
    metrics = {
        "dataset": args.dataset,
        "test": regression_metrics(y_true, y_pred),
        "test_vs_capped_truth": regression_metrics(capped_truth, y_pred),
        "test_single_members": [regression_metrics(y_true, p) for p in member_preds],
        "validation_all_windows": regression_metrics(w["y_val"], val_pred),
        "validation_last_window": regression_metrics(
            va_last_true, predict(ensemble, X_va_last, device, target_scale, clip_max=rul_clip)),
        "members": member_info,
        "hyperparameters": best_hp,
        "runtime_seconds": round(time.time() - t_start, 1),
        "device": dev_name,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plots.pred_vs_true(y_true, y_pred, out / "test_pred_vs_true.png",
                       f"{args.dataset} test set: predicted vs true RUL")
    plots.engines_sorted(y_true, y_pred, out / "test_engines_sorted.png",
                         f"{args.dataset} test set: RUL per engine")
    test_traj = D.add_test_rul(test_df, true_rul)
    longest = test_traj.groupby("engine_id")["cycle"].max().nlargest(4).index.sort_values()
    curves = []
    for eid in longest:
        g = test_traj[test_traj["engine_id"] == eid]
        Xw, yw, _ = D.make_train_windows(g, features, seq_len)
        curves.append(dict(engine_id=int(eid), cycle=g["cycle"].to_numpy()[-len(yw):],
                           true=np.minimum(yw, rul_clip) if rul_clip else yw,
                           pred=predict(ensemble, Xw, device, target_scale, clip_max=rul_clip)))
    plots.trajectories(curves, out / "rul_trajectories.png",
                       f"{args.dataset}: predicted RUL over the life of the four longest test engines")

    config = dict(vars(args))
    config.update(features=features, n_features=len(features), target_scale=target_scale,
                  train_engines=train_ids, val_engines=val_ids, model_config=ensemble.config,
                  best_hyperparameters=best_hp, optuna_trials=trials_info,
                  n_windows={"train": len(w["X_train"]), "val": len(w["X_val"]), "test": len(X_te)},
                  torch_version=torch.__version__, created=datetime.now().isoformat(timespec="seconds"))
    (out / "training_config.json").write_text(json.dumps(config, indent=2, default=str))

    t, c = metrics["test"], metrics["test_vs_capped_truth"]
    print(f"\nTEST {args.dataset} ({t['n']} engines, ensemble of {len(members)}): RMSE {t['rmse']:.2f} | "
          f"MAE {t['mae']:.2f} | NASA score {t['nasa_score']:.1f}   [capped truth: RMSE {c['rmse']:.2f}]")
    print(f"Total time {metrics['runtime_seconds']:.0f}s. Artifacts written to {out}")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except D.DatasetNotFoundError as e:
        sys.exit(f"ERROR: {e}")
