"""Implementation of a simple multilayer perceptron network."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import flwr as fl

class Net(nn.Module):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 24)
        self.fc2 = nn.Linear(24, num_classes)
        self._num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
