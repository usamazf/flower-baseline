"""Evaluation function to test the model performance."""

from typing import Tuple

import torch
import torch.nn as nn

def evaluate(
        model,
        testloader: torch.utils.data.DataLoader,
        device: str,
    ) -> Tuple[float, float]:
    
    """Validate the model on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy