#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     G L O B A L     L I B R A R I E S                        #
#                                                                            #
#----------------------------------------------------------------------------#
import argparse
import timeit

import torch
import torchvision

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

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
DEVICE = torch.device("cuda:0" if glb.USE_GPU else "cpu")

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   extend the flower client base class to implement new clients.            #
#                                                                            #
#****************************************************************************#
class PyTorchClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        trainset: torchvision.datasets.CIFAR10,
        testset: torchvision.datasets.CIFAR10,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")
        
        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")
        
        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        
        # Set model parameters
        self.model.set_weights(weights)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
        )
        modules.pt_train(self.model, trainloader, epochs=epochs, device=DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        
        # check if quantization is requested
        if glb.QUANTIZE:
            weights_prime: Weights = modules.quantize(
                weights=weights_prime, 
                bits=glb.Q_BITS
            )
        
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        loss, accuracy = modules.pt_test(self.model, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), accuracy=float(accuracy)
        )

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   Load data, create and start PyTorchClient.                               #
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
    model = models.load_model(model_name=glb.MODEL, framework="PT")
    model.to(DEVICE)
    trainset, testset = datasets.load_data(dataset_name=glb.DATASET, framework="PT")

    # Start client
    client = PyTorchClient(args.cid, model, trainset, testset)
    try:
        fl.client.start_client(args.server_address, client)
    except:
        print("Either something went wrong or server finished execution!!")


if __name__ == "__main__":
    main()
