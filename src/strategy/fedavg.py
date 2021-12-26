#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     G L O B A L     L I B R A R I E S                        #
#                                                                            #
#----------------------------------------------------------------------------#
from typing import Callable, Dict, List, Optional, Tuple

from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
from configs import globals as glb
import modules

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   extend the strategy base class to implement the federated averaging algorithm by self.      #
#                                                                                               #
#***********************************************************************************************#
class FederatedAverage(Strategy):
    """Configurable FedAvg strategy implementation."""
    
    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        accept_failures: bool = True,
        dummy_model = None,
     ) -> None:
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        # a dummy model used to determine dimensions of weights vector if quantization is used
        self.dummy_model = dummy_model
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
        
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients
    
    def configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
  
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            print(failures)
            return None
        # Convert results
        if glb.QUANTIZE:
            weights_results = [
                (modules.dequantize(self.dummy_model, parameters_to_weights(fit_res.parameters), glb.Q_BITS), fit_res.num_examples)
                for client, fit_res in results
            ]
        else:
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
        return aggregate(weights_results)

    def on_conclude_round(
        self, rnd: int, loss: Optional[float], acc: Optional[float]
    ) -> bool:
        """Always continue training."""
        return True

    def configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results."""
        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        return weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss, evaluate_res.accuracy)
                for _, evaluate_res in results
            ]
        )
    
    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Evaluate the current model weights."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        return self.eval_fn(weights)
