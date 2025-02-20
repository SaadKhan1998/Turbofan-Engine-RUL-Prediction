
import matplotlib.pyplot as plt
import torch

# Save loss plots to a file
def save_plot(losses, output_path, plot_title):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Save hyperparameters to a text file
def save_hyperparameters(hyperparams, output_path):
    with open(output_path, 'w') as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")

# Save model state dictionary to a file
def save_model_dict(model, output_path):
    torch.save(model.state_dict(), output_path)

# Saves the PyTorch model to a file
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Loads the PyTorch model from a file.
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

# save best updated model
def save_best_model(model, val_loss, best_loss, path):
    if val_loss < best_loss:
        print(f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model...")
        save_model(model, path)
        best_loss = val_loss
    return best_loss