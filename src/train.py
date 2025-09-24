# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from model import MakeSense
from config import *

def train():
    train_loader = get_dataloader(DATA_PATH, BATCH_SIZE)
    model = MakeSense(input_dim=10, output_dim=2).to(DEVICE)  # Replace dims
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()
