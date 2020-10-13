#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     G L O B A L     L I B R A R I E S                        #
#                                                                            #
#----------------------------------------------------------------------------#
import argparse
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision

import flwr as fl
from strategy.fedavg import FederatedAverage

#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
from configs import globals as glb
import modules
import models
import datasets

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
#   Start server and train five rounds.                                      #
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
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host", type=str, help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    _, testset = datasets.load_data(dataset_name=glb.DATASET, framework="PT")

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = FederatedAverage(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
        dummy_model = models.load_model(model_name=glb.MODEL, framework="PT"),        
    )
    #strategy = fl.server.strategy.DefaultStrategy(
    #    fraction_fit=args.sample_fraction,
    #    min_fit_clients=args.min_sample_size,
    #    min_available_clients=args.min_num_clients,
    #    eval_fn=get_eval_fn(testset),
    #    on_fit_config_fn=fit_config,
    #)
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        args.server_address, server, config={"num_rounds": args.rounds},
    )


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(32),
        "learning_rate": str(0.001),
    }
    return config


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = models.load_model(glb.MODEL)
        model.set_weights(weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        # using pytorch for central evaluation, can be tensorflow as well
        return modules.pt_test(model, testloader, device=DEVICE) 

    return evaluate


if __name__ == "__main__":
    main()
