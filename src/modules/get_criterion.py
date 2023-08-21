"""Help function to get requested criterion / loss function."""

import torch.nn as nn

def get_criterion(
        criterion_str: str,    
    ) -> None:
    """Get requested criterion / loss function."""
    assert criterion_str in ["CROSSENTROPY", "NLLL"], f"Invalid criterion {criterion_str} requested."

    if criterion_str == "CROSSENTROPY":
        return nn.CrossEntropyLoss()
    elif criterion_str == "NLLL":
        return nn.NLLLoss()
    else:
        raise Exception(f"Invalid criterion {criterion_str} requested.")
