
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Load CMAPSS data
def load_cmapss(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None, engine='python')
    data.dropna(axis=1, inplace=True)
    columns = ['engine_id', 'cycle'] + [f'op_set_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    data.columns = columns
    return data

# Create sequences for the Transformer model
def create_sequences(data, seq_length, feature_columns):
    sequences = []
    labels = []
    for engine_id in data['engine_id'].unique():
        engine_data = data[data['engine_id'] == engine_id]
        for i in range(len(engine_data) - seq_length):
            seq = engine_data.iloc[i:i+seq_length][feature_columns].values
            label = engine_data.iloc[i+seq_length]['RUL']
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

# Createing sequences from the data for inference
def create_sequences_for_inference(data, seq_length, feature_columns):

    sequences = []
    engine_ids = []

    for engine_id in data['engine_id'].unique():
        engine_data = data[data['engine_id'] == engine_id]
        for i in range(len(engine_data) - seq_length + 1):
            seq = engine_data.iloc[i:i + seq_length][feature_columns].values
            sequences.append(seq)
            engine_ids.append(engine_id)

    return np.array(sequences), engine_ids


# Custom dataset class
class CMAPSSDataset(Dataset):
    def __init__(self, X, y=None):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]  # Return only features during inference