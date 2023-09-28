"""Module to run the Federated Learning server specified by experiment configurations."""

import ntpath
import argparse
from typing import List, Tuple, Union

import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

from exp_manager import ExperimentManager
from modules import server
import configs
import strategy

def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help="gRPC server address (default: [::]:8080)",
    )
    parser.add_argument(
        "--config_file",
        type = str,
        required = True,
        help="Configuration file to use (no default)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()
    user_configs = configs.parse_configs(args.config_file)
    
    # Fetch stats and store them locally?
    exp_config = ntpath.basename(args.config_file)
    exp_manager = ExperimentManager(experiment_id=exp_config[:-5], hyperparameters=user_configs)

    # Create strategy
    agg_strat = strategy.get_strategy(user_configs)

    # create a client manager
    client_manager = SimpleClientManager()

    # create a server
    custom_server = server.Server(
        client_manager=client_manager, 
        strategy=agg_strat,
        experiment_manager=exp_manager,
        early_stop=user_configs["SERVER_CONFIGS"]["EARLY_STOP"]
    )

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"]),
        server=custom_server,
    )

    exp_manager.save_to_disc(user_configs["SERVER_CONFIGS"]["LOG_RESULT_PATH"], exp_config[:-5])

if __name__ == "__main__":
    main()
