
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import os
from dataset import load_cmapss, create_sequences, CMAPSSDataset
from utils import save_plot, save_hyperparameters, save_model_dict
from optimizer import run_optimization, create_results_folder
from training import preprocess_test_data


if __name__ == "__main__":

    # check if gpu available for faster computing 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data -- // Specify the paths // 
    train_file = 'C:/Users/imran/OneDrive/Desktop/group_19_assignment/group_19_assignment/CMAPSS/train_FD003.txt'
    rul_file = 'C:/Users/imran/OneDrive/Desktop/group_19_assignment/group_19_assignment/CMAPSS/RUL_FD003.txt'
    test_file = 'C:/Users/imran/OneDrive/Desktop/group_19_assignment/group_19_assignment/CMAPSS/test_FD003.txt'
    
    train_data = load_cmapss(train_file)
    max_cycle = train_data.groupby('engine_id')['cycle'].max()
    train_data = train_data.merge(max_cycle.rename('max_cycle'), on='engine_id')
    train_data['RUL'] = train_data['max_cycle'] - train_data['cycle']

    scaler = MinMaxScaler()
    feature_columns = [col for col in train_data.columns if 'sensor_' in col or 'op_set_' in col]
    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])

    # Prepare sequences
    seq_length = 30
    input_dim = len(feature_columns)
    hidden_dim = 64
    n_heads = 4
    n_layers = 2

    X, y = create_sequences(train_data, seq_length, feature_columns)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = CMAPSSDataset(X_train, y_train)
    val_dataset = CMAPSSDataset(X_val, y_val)
    
    # Load true RUL values for test data
    test_labels = pd.read_csv(rul_file, header=None).values.squeeze()
    # Preprocess test data
    
    test_dataset, engine_ids, sequence_labels = preprocess_test_data(test_file, scaler, feature_columns, seq_length, test_labels)
 

    # Create results folder
    results_folder = create_results_folder()["base"]

    # Run optimization
    n_trials = 2
    best_params = run_optimization(
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
        n_trials=n_trials
    )

    print("Best hyperparameters:", best_params)
    
    # Assume train_losses, val_losses, model, and hyperparameters are available
    train_losses = [0.5, 0.4, 0.3]  # Example data
    val_losses = [0.6, 0.5, 0.35]  # Example data
    hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 2
    }  # Example data

    # Assuming 'model' is your trained PyTorch model
    model = torch.nn.Linear(10, 1)  # Example model

    # Create results folder
    folders = create_results_folder()

    # Save plots
    save_plot(train_losses, os.path.join(folders["training"], "train_loss_plot.png"), "Training Loss")
    save_plot(val_losses, os.path.join(folders["validation"], "val_loss_plot.png"), "Validation Loss")

    # Save hyperparameters
    save_hyperparameters(hyperparameters, os.path.join(folders["base"], "hyperparameters.txt"))

    # Save model state dict
    save_model_dict(model, os.path.join(folders["base"], "model_state.pth"))

    print(f"Results saved in folder: {folders['base']}")
