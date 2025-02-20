
import torch
import pandas as pd
import numpy as np
from dataset import CMAPSSDataset
from utils import save_best_model

def preprocess_test_data(test_file, scaler, feature_columns, seq_length, test_labels):

    test_data = pd.read_csv(test_file, delim_whitespace=True, header=None)
    test_data.columns = ['engine_id', 'cycle'] + [f'op_set_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    sequences = []
    engine_ids = []
    sequence_labels = []  # True RUL for each sequence

    for idx, engine_id in enumerate(test_data['engine_id'].unique()):
        engine_data = test_data[test_data['engine_id'] == engine_id]
        true_rul = test_labels[idx]  # RUL for this engine
        for i in range(len(engine_data) - seq_length + 1):
            seq = engine_data.iloc[i:i + seq_length][feature_columns].values
            sequences.append(seq)
            engine_ids.append(engine_id)
            sequence_labels.append(true_rul)  # Assign the engine's RUL to all its sequences

    # Create a dataset with the sequences (labels are placeholders for inference)
    test_dataset = CMAPSSDataset(np.array(sequences), None)

    return test_dataset, engine_ids, sequence_labels

def train_model(model,train_loader, val_loader, test_loader, test_labels, engine_ids, criterion, optimizer, device, epochs, 
save_path):
  
    best_loss = float('inf')
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_absolute_error = 0  # For accuracy calculation
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
                val_absolute_error += torch.sum(torch.abs(outputs.squeeze() - y_batch)).item()  # Accumulate absolute errors
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Calculate accuracy as Mean Absolute Error (MAE)
        val_mae = val_absolute_error / len(val_loader.dataset)
        val_accuracies.append(val_mae)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        # Save the best model
        best_loss = save_best_model(model, val_loss, best_loss, save_path)
        
    # Inference phase
    print("Starting inference on test data...")
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.squeeze().item())

    # Create a DataFrame for predictions and ground truth
    results_df = pd.DataFrame({
        "Engine_ID": engine_ids,
        "Ground_Truth_RUL": test_labels,
        "Predicted_RUL": predictions
    })

    print("Inference completed.")

    return train_losses, val_losses, val_accuracies, results_df
