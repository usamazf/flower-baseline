#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
from typing import Tuple, cast

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models

import flwr as fl

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out the program              #
#                                                                            #
#----------------------------------------------------------------------------#


#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'.                    #
#                                                                            #
#****************************************************************************#
class Net():
    
    def __init__(self):
        self.model = self.build_model()
            
    def build_model(
        self, input_shape: Tuple[int, int, int] = (32, 32, 3)
    ) -> tf.keras.Model:
        # Build model layers
        model = models.Sequential()
        # Convolutional Layers
        model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # Flatten Layer
        model.add(layers.Flatten())
        # Fully connected layer
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(10))
        # compile the model
        model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        # return newly built model
        return model
    
    def fit(self, x_train, y_train, batch_size, epochs) -> None:
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def evaluate(
        self, x_test, y_test, batch_size, verbose
    ) -> Tuple[int, float, float]:
        return self.model.evaluate(
            x_test, y_test, batch_size=batch_size, verbose=verbose
        )
    
    def get_weights(self) -> fl.common.Weights:
        # fetch latest weights
        weights = self.model.get_weights()
        # transpose weights
        t_weights = []
        for w in weights:
            t_weights.append(np.transpose(w))
        # return by casting to flower weights format
        return cast(fl.common.Weights, t_weights)
    
    def set_weights(self, weights: fl.common.Weights) -> None:
        t_weights = []
        # transpose into acceptable format
        for w in weights:
            t_weights.append(np.transpose(w))
        # set the weights back to model
        self.model.set_weights(t_weights)
    