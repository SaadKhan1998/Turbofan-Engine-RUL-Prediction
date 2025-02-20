
import optuna
from torch.utils.data import DataLoader
import torch
import os
from datetime import datetime
import torch.nn as nn
from utils import save_plot, save_hyperparameters, save_model_dict

def create_results_folder(base_path="results"):

    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = os.path.join(base_path, f"run_{timestamp}")

    # Create subfolders for training, validation, and inference
    train_folder = os.path.join(base_folder, "training")
    val_folder = os.path.join(base_folder, "validation")
    inference_folder = os.path.join(base_folder, "inference")

    # Ensure all folders are created
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(inference_folder, exist_ok=True)

    return {
        "base": base_folder,
        "training": train_folder,
        "validation": val_folder,
        "inference": inference_folder
    }

# Initialize a cache to store tried hyperparameter combinations
tested_hyperparameters = set()

def objective(trial, train_dataset, val_dataset, test_dataset, sequence_labels, engine_ids, input_dim, seq_length, n_heads, hidden_dim, n_layers, device, results_folder):
    global tested_hyperparameters

    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    print(f"Current learning rate: {learning_rate}")
    print(f"Current Batch size: {batch_size}")

    # Create a tuple of the current hyperparameters
    hyperparameter_config = (learning_rate, batch_size)

    # Check if the configuration has already been tested
    if hyperparameter_config in tested_hyperparameters:
        print(f"Pruning trial {trial.number}: Duplicate hyperparameters {hyperparameter_config}")
        raise optuna.exceptions.TrialPruned()

    # Add the current configuration to the cache
    tested_hyperparameters.add(hyperparameter_config)

    # Update data loaders with suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model, criterion, and optimizer
    from model import Model
    from training import train_model

    model = Model(input_dim=input_dim, seq_length=seq_length, n_heads=n_heads, hidden_dim=hidden_dim, n_layers=n_layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Save results for this trial
    trial_folder = create_results_folder(base_path=results_folder)
    epochs = 2
    save_path = f"{trial_folder['base']}/best_model.pth"

    # Train the model
    train_losses, val_losses, val_accuracies, results_df = train_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        sequence_labels, 
        engine_ids, 
        criterion, 
        optimizer, 
        device, 
        epochs, 
        save_path=save_path
    )

    # Save training, validation, and inference results
    save_plot(train_losses, f"{trial_folder['training']}/train_loss_plot.png", "Training Loss")
    save_plot(val_losses, f"{trial_folder['validation']}/val_loss_plot.png", "Validation Loss")
    save_plot(val_accuracies, f"{trial_folder['validation']}/val_accuracy_plot.png", "Validation Accuracy")
    save_hyperparameters({"learning_rate": learning_rate, "batch_size": batch_size}, f"{trial_folder['base']}/hyperparameters.txt")
    save_model_dict(model, f"{trial_folder['base']}/model_state.pth")

    # Save inference results to the inference folder
    inference_results_path = os.path.join(trial_folder['inference'], "predicted_rul_with_ground_truth.csv")
    results_df.to_csv(inference_results_path, index=False)
    print(f"Inference results saved to {inference_results_path}")

    # Return the final validation loss as the metric to minimize
    return val_losses[-1]

def run_optimization(
    train_dataset, 
    val_dataset, 
    test_dataset, 
    sequence_labels, 
    engine_ids, 
    input_dim, 
    seq_length, 
    n_heads, 
    hidden_dim, 
    n_layers, 
    device, 
    results_folder, 
    n_trials=2
):
    study = optuna.create_study(direction="minimize")  # We want to minimize validation loss
    study.optimize(
        lambda trial: objective(
            trial, 
            train_dataset, 
            val_dataset, 
            test_dataset, 
            sequence_labels, 
            engine_ids, 
            input_dim, 
            seq_length, 
            n_heads, 
            hidden_dim, 
            n_layers, 
            device, 
            results_folder
        ),
        n_trials=n_trials
    )

    # Save best hyperparameters
    best_hyperparams_path = f"{results_folder}/best_hyperparameters.txt"
    save_hyperparameters(study.best_params, best_hyperparams_path)

    return study.best_params
