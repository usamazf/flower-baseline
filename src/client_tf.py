#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     G L O B A L     L I B R A R I E S                        #
#                                                                            #
#----------------------------------------------------------------------------#
import argparse
import timeit

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
class TensorflowClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""
    # need to implement this
    pass

if __name__ == "__main__":
    main()
