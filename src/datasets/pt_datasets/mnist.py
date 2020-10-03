#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
from typing import Tuple

import torchvision
import torchvision.transforms as transforms

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out the program              #
#                                                                            #
#----------------------------------------------------------------------------#
DATA_ROOT = "./data/mnist"

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   load and return the training and testing sets of cifar-10 dataset.       #
#                                                                            #
#****************************************************************************#
def load_mnist() -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.1307,))]
    )

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    # Return the datasets
    return trainset, testset
