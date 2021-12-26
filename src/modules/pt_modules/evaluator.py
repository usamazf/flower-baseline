#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
from typing import Tuple

import torch
import torch.nn as nn

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   validation function to evaluate the model on entire test set.            #
#                                                                            #
#****************************************************************************#
def test(net, 
         testloader: torch.utils.data.DataLoader, 
         device: torch.device,
        ) -> Tuple[float, float]:
    
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, {"accuracy": accuracy}