#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   return the desired model.                                                #
#                                                                            #
#****************************************************************************#
def get_strategy(
        strategy_name,
        fraction_fit, 
        min_fit_clients, 
        min_available_clients, 
        eval_fn,
        on_fit_config_fn,
        dummy_model=None,
        quantize=False,
        quantize_bits=64):
    
    if strategy_name == "FedAvg":
        # import the model from the local directory
        if quantize:
            from .strats.fedavg import FederatedAverage
            return FederatedAverage(
                            fraction_fit=fraction_fit,
                            min_fit_clients=min_fit_clients,
                            min_available_clients=min_available_clients,
                            eval_fn=eval_fn,
                            on_fit_config_fn=on_fit_config_fn)
        else:
            from .strats.fedavg_qt import FederatedAverage_Q
            return FederatedAverage_Q(
                            fraction_fit=fraction_fit,
                            min_fit_clients=min_fit_clients,
                            min_available_clients=min_available_clients,
                            eval_fn=eval_fn,
                            on_fit_config_fn=on_fit_config_fn,
                            dummy_model=dummy_model,
                            quantize_bits=quantize_bits)

 