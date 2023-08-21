"""Training function to train the model for given number of epochs."""

import torch
import torch.nn as nn

def train(model,
          trainloader: torch.utils.data.DataLoader,
          epochs: int,
          device: str,  # pylint: disable=no-member
          learning_rate: float,
          criterion,
          optimizer,
         ) -> None:
    """Train the model."""
    # Define loss and optimizer
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 0 and i > 0:  # print every 500 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0