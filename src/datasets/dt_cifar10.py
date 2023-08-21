"""A function to load the CIFAR-10 dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_cifar10(data_root) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load CIFAR-10 (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    # Initialize Datasets. CIFAR-10 will automatically download if not present
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    
    # Return the datasets
    return trainset, testset