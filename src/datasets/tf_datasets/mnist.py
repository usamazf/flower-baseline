#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
from typing import Tuple

import os
import numpy as np

import tensorflow as tf

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out the program              #
#                                                                            #
#----------------------------------------------------------------------------#
dirname = "./data/mnist/"

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   load and return the training and testing sets of cifar-10 dataset.       #
#                                                                            #
#****************************************************************************#
def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    
    # Loading datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    
    # Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Return dataset
    return (x_train, y_train), (x_test, y_test)
