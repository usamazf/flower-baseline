"""A function to load the Fashion MNIST digit dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_fmnist(data_root) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load Fashion MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),    
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    # Return the datasets
    return trainset, testset