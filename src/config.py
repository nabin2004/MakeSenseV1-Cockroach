# src/config.py
import torch

# Paths
DATA_PATH = "../data/processed"
OUTPUT_PATH = "../outputs"

# Hyperparameters
SEED = 42
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
