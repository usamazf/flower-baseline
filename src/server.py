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
from strategy import get_strategy

#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
import modules
import models
import datasets

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out the program              #
#                                                                            #
#----------------------------------------------------------------------------#
DEFAULT_SERVER_ADDRESS = "127.0.0.1:8080"

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
        "--model",
        type=str,
        default="simple-cnn",
        help="Model to use for training (default: simple-cnn)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar-10",
        help="Dataset to use fro training (default: cifar-10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Device to run the model on (default: CPU)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        help="Aggregation strategy (default: FedAvg)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of local epochs to run on each client before aggregation (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to be used by each worker (default: 32)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate to be used by each worker (default: 0.001)",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=False,
        help="Use quantization (default: False)",
    )
    parser.add_argument(
        "--quantize_bits",
        type=int,
        default=64,
        help="Quantization bits (default: 64)",
    )
    parser.add_argument(
        "--log_host", type=str, help="Logserver address (no default)",
    )
    args = parser.parse_args()
    
    # check for runnable device
    DEVICE = torch.device("cuda:0" if args.device == "GPU" else "cpu")
    
    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    _, testset = datasets.load_data(dataset_name=args.dataset, framework="PT")

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    aggregation_strategy = get_strategy(
        strategy_name = args.strategy,
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(model=args.model, testset=testset, device=DEVICE),
        on_fit_config_fn=get_fit_config_fn(args),
        dummy_model = models.load_model(model_name=args.model, framework="PT"),        
        quantize = args.quantize,
        quantize_bits=args.quantize_bits,
    )
    
    server = fl.server.Server(client_manager=client_manager, strategy=aggregation_strategy)

    # Run server
    fl.server.start_server(
        args.server_address, server, config={"num_rounds": args.rounds},
    )

def get_fit_config_fn(args: Dict) -> Callable[[int], Optional[Dict[str, str]]]:
    """Return a callable configuration function to fetch fit configurations."""
    # create on fit configuration function
    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
            "learning_rate": str(args.learning_rate),
            "quantize": str(args.quantize),
            "quantize_bits": str(args.quantize_bits),
        }
        return config
    
    return fit_config

def get_eval_fn(
    testset: torchvision.datasets,
    model: str,
    device: str,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, Dict]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, Dict]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = models.load_model(model)
        model.set_weights(weights)
        model.to(device)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        # using pytorch for central evaluation, can be tensorflow as well
        return modules.pt_test(model, testloader, device=device) 

    return evaluate


if __name__ == "__main__":
    main()
