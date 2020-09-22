#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
from typing import Tuple, cast

import numpy as np
import tensorflow as tf

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
        self, seed: int = 28, input_shape: Tuple[int, int, int] = (32, 32, 3)
    ) -> tf.keras.Model:
        # Kernel initializer
        kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed)

        # Build model layers
        inputs = tf.keras.layers.Input(shape=input_shape)
        # Convolutional Layer
        layers = tf.keras.layers.Conv2D(
            6, 
            kernel_size=(5,5), 
            kernel_initializer=kernel_initializer, 
            activation='relu'
        )(inputs)
        # Max Pooling Layer
        layers = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), 
            strides=(2, 2)
        )(layers)
        # Convolutional Layer
        layers = tf.keras.layers.Conv2D(
            16, 
            kernel_size=(5,5), 
            kernel_initializer=kernel_initializer, 
            activation='relu'
        )(layers)
        # Max Pooling Layer
        layers = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), 
            strides=(2, 2)
        )(layers)
        # Reshape Layer
        layers = tf.keras.layers.Flatten()(layers)
        # FC 1
        layers = tf.keras.layers.Dense(
            120, kernel_initializer=kernel_initializer, activation="relu"
        )(layers)
        # FC 2
        layers = tf.keras.layers.Dense(
            84, kernel_initializer=kernel_initializer, activation="relu"
        )(layers)
        # Output Layer
        outputs = tf.keras.layers.Dense(
            10, kernel_initializer=kernel_initializer, activation="softmax"
        )(layers)

        # build model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # compile the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )

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
    