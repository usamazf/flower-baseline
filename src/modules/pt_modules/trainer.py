#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
import torch
import torch.nn as nn

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   function used to train the model on the current dataset.                 #
#                                                                            #
#****************************************************************************#
def train(net,
          trainloader: torch.utils.data.DataLoader,
          epochs: int,
          device: torch.device,  # pylint: disable=no-member
          learning_rate: float,
         ) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 500 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
