"""A function to load the desired aggregation strategy."""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

import flwr as fl

import sys
import os
sys.path.append(os.path.abspath("src"))
import datasets
import models
import modules


def get_strategy(user_configs: dict):
    # Check what device to use on 
    # server side to run the computations
    run_device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if user_configs["SERVER_CONFIGS"]["RUN_DEVICE"] == "auto" \
            else user_configs["SERVER_CONFIGS"]["RUN_DEVICE"]
    
    # Check wether to evaluate the global
    # model on the server side or not
    eval_fn = None
    if user_configs["SERVER_CONFIGS"]["EVALUATE_SERVER"]:
        # Load evaluation data
        _, testset = datasets.load_data(
            dataset_name=user_configs["DATASET_CONFIGS"]["DATASET_NAME"],
            dataset_path=user_configs["DATASET_CONFIGS"]["DATASET_PATH"]
        )

        eval_fn = get_evaluate_fn(
            testset=testset,
            model_configs=user_configs["MODEL_CONFIGS"],
            device=run_device
        )

    # Build the fit config function
    fit_config_fn = get_fit_config_fn(
        local_epochs=user_configs["CLIENT_CONFIGS"]["LOCAL_EPCH"],
        local_batchsize=user_configs["CLIENT_CONFIGS"]["BATCH_SIZE"],
        learning_rate=user_configs["CLIENT_CONFIGS"]["LEARN_RATE"],
        optimizer_str=user_configs["CLIENT_CONFIGS"]["OPTIMIZER"],
        criterion_str=user_configs["CLIENT_CONFIGS"]["CRITERION"],
    )
    
    # Create an instance of the 
    # desired aggregation strategy
    if user_configs["SERVER_CONFIGS"]["AGGREGATE_STRAT"] == "FEDAVG":
        from .strategies.normal_fedavg import FederatedAverage
        stratgy = FederatedAverage(
            fraction_fit=user_configs["SERVER_CONFIGS"]["SAMPLE_FRACTION"],
            min_fit_clients=user_configs["SERVER_CONFIGS"]["MIN_SAMPLE_SIZE"],
            min_available_clients=user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"],
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
        )
        return stratgy
    else:
        raise Exception(f"Invalid aggregation strategy {user_configs['SERVER_CONFIGS']['AGGREGATE_STRAT']} requested.")

def get_fit_config_fn(
        local_epochs, 
        local_batchsize, 
        learning_rate, 
        optimizer_str,
        criterion_str,
    ):
    def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config: Dict[str, fl.common.Scalar] = {
            "epoch_global": str(server_round),
            "epochs": str(local_epochs),
            "batch_size": str(local_batchsize),
            "learning_rate": str(learning_rate),
            "optimizer": optimizer_str,
            "criterion": criterion_str,
        }
        return config
    return fit_config


def get_evaluate_fn(
    testset: Dataset,
    model_configs: dict,
    device: str,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, 
            weights: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = models.load_model(model_configs=model_configs)
        model.set_weights(weights)
        model.to(device)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = modules.evaluate(model, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate
