
import torch.nn as nn

# Transformer Encoder Class
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, n_heads, hidden_dim, n_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True  # Batch-first input
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        return self.transformer(x)

# MLP-Decoder Class
class MLPDecoder(nn.Module):
    def __init__(self, input_dim, seq_length):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * seq_length, 128),  # Flatten the input
            nn.ReLU(),
            nn.Linear(128, 1)  # Predict a single RUL value
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)  
        return self.decoder(x)

# Combined Model Class
class Model(nn.Module):
    def __init__(self, input_dim, seq_length, n_heads, hidden_dim, n_layers):
        super(Model, self).__init__()
        self.encoder = TransformerEncoder(
            input_dim=input_dim, 
            n_heads=n_heads, 
            hidden_dim=hidden_dim, 
            n_layers=n_layers
        )
        self.decoder = MLPDecoder(input_dim=input_dim, seq_length=seq_length)

    def forward(self, x):
        encoded = self.encoder(x)  # Transformer Encoder output
        decoded = self.decoder(encoded)  # MLP Decoder output
        return decoded
