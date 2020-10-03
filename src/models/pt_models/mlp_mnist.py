#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import flwr as fl

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out the program              #
#                                                                            #
#----------------------------------------------------------------------------#


#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'.                    #
#                                                                            #
#****************************************************************************#
class Net(nn.Module):
    
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 24)
        self.fc2 = nn.Linear(24, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
