"""Help function to get requested criterion / loss function."""

import torch
import torch.nn as nn

def get_optimizer(
        optimizer_str: str,
        local_model: nn.Module,
        learning_rate: float,
    ) -> None:
    """Get requested criterion / loss function."""
    assert optimizer_str in ["SGD", "ADAM"], f"Invalid optimizer {optimizer_str} requested."

    if optimizer_str == "SGD":
        return nn.CrossEntropyLoss()
    elif optimizer_str == "ADAM":
        return torch.optim.Adam(local_model.parameters(), lr=learning_rate)
    else:
        raise Exception(f"Invalid criterion {optimizer_str} requested.")
