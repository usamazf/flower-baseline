#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     G L O B A L     L I B R A R I E S                        #
#                                                                            #
#----------------------------------------------------------------------------#
import argparse
import timeit
from typing import Dict, Tuple, cast

import tensorflow
import numpy as np

import flwr as fl
from flwr.common import Weights

#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
from configs import globals as glb
import models
import datasets
import modules

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out the program              #
#                                                                            #
#----------------------------------------------------------------------------#
DEFAULT_SERVER_ADDRESS = "127.0.0.1:8080"
#DEVICE = torch.device("cuda:0" if glb.USE_GPU else "cpu")

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   extend the flower client base class to implement new clients.            #
#                                                                            #
#****************************************************************************#
class TfKerasClient(fl.client.KerasClient):
    """Flower KerasClient implementing CIFAR-10 image classification using Tensorflow."""
    
    def __init__(
        self,
        cid: str,
        model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
    ):
        self.cid = cid
        self.model = model
        self.x_train, self.y_train = xy_train
        self.x_test, self.y_test = xy_test

    def get_weights(self) -> Weights:
        return self.model.get_weights()

    def fit(self, weights: Weights, config: Dict[str, str]) -> Tuple[Weights, int, int]:
        # Use provided weights to update local model
        self.model.set_weights(weights)

        # Train the local model using local dataset
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
        )

        # Return the refined weights and the number of examples used for training
        return self.model.get_weights(), len(self.x_train), len(self.x_train)

    def evaluate(
        self, weights: Weights, config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Update local model and evaluate on local dataset
        self.model.set_weights(weights)
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test, batch_size=len(self.x_test), verbose=2
        )

        # Return number of evaluation examples and evaluation result (loss/accuracy)
        return len(self.x_test), float(loss), float(accuracy)        
        
#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   Load data, create and start PyTorchClient.                                 #
#                                                                            #
#****************************************************************************#
def main() -> None:
    
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host", type=str, help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    model = models.load_model(model_name=glb.MODEL, framework="TF")
    xy_train, xy_test = datasets.load_data(dataset_name=glb.DATASET, framework="TF")

    # Start client
    keras_client = TfKerasClient(args.cid, model, xy_train, xy_test)
    client = fl.client.keras_client.KerasClientWrapper(keras_client)
    
    #try:
    fl.client.start_client(args.server_address, client)
    #except:
    #    print("Either something went wrong or server finished execution!!")


if __name__ == "__main__":
    main()
