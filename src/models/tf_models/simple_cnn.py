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
#   implementation of the actual model to make it easier for testing.        #
#                                                                            #
#****************************************************************************#
class SimpleCNN(tf.keras.Model):
    
    def __init__(self, input_shape: Tuple[int, int, int], seed: int):
        super(SimpleCNN, self).__init__()
        # define initializer
        #kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed)
        # build model layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=6, 
            kernel_size=(5, 5),
            strides=(1,1),
            padding='valid', 
            #kernel_initializer=kernel_initializer, 
            activation='relu', 
            input_shape=input_shape
        )
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(5, 5),
            strides=(1,1),
            padding='valid', 
            #kernel_initializer=kernel_initializer, 
            activation='relu'
        )
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(
            units=120, 
            activation='relu',
            #kernel_initializer=kernel_initializer
        )
        self.fc2 = tf.keras.layers.Dense(
            units=84, 
            activation='relu',
            #kernel_initializer=kernel_initializer
        )
        self.fc3 = tf.keras.layers.Dense(
            units=10, 
            #kernel_initializer=kernel_initializer
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def extract_weights(self, style):
        weights = []
        # check weight style
        if style=="pytorch":
            # Layer 1
            temp = self.conv1.get_weights()
            weights.append(temp[0].transpose(3,2,0,1))
            weights.append(temp[1].transpose())
            # Layer 2
            temp = self.conv2.get_weights()
            weights.append(temp[0].transpose(3,2,0,1))
            weights.append(temp[1].transpose())
            # Layer 3
            temp = self.fc1.get_weights()
            weights.append(temp[0].transpose())
            weights.append(temp[1].transpose())
            # Layer 4
            temp = self.fc2.get_weights()
            weights.append(temp[0].transpose())
            weights.append(temp[1].transpose())
            # Layer 5
            temp = self.fc3.get_weights()
            weights.append(temp[0].transpose())
            weights.append(temp[1].transpose())
        
        elif style=="tensorflow":
            weights = self.get_weights()
        
        # return extracted weights
        return weights
        
    def setup_weights(self, weights, style):
        # check weight style
        if style=="pytorch":
            self.conv1.set_weights( 
                [weights[0].transpose(2,3,1,0), weights[1].transpose()]
            )
            self.conv2.set_weights( 
                [weights[2].transpose(2,3,1,0), weights[3].transpose()]
            )
            self.fc1.set_weights(
                [weights[4].transpose(), weights[5].transpose()]
            )
            self.fc2.set_weights(
                [weights[6].transpose(), weights[7].transpose()]
            )
            self.fc3.set_weights(
                [weights[8].transpose(), weights[9].transpose()]
            )
        
        elif style=="tensorflow":
            self.set_weights(weights)

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'.                    #
#                                                                            #
#****************************************************************************#
class Net():
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (32, 32, 3), 
        seed: int = 42,
    ):
        self.model = self.build_model(seed=seed, input_shape=input_shape)
    
    def initialize_weights(self, model, input_shape):
        random_sample = np.random.random_sample((1,)+input_shape)
        model.predict(random_sample)
            
    def build_model(self, seed, input_shape) -> tf.keras.Model:
        # create model instance
        model = SimpleCNN(input_shape=input_shape, seed=seed)
        # create optimizer
        optim = tf.keras.optimizers.SGD(learning_rate=self.lr_scheduler, momentum=0.9)
        # compile this model
        model.compile(optimizer=optim,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        # initialize model weights
        self.initialize_weights(model, input_shape)
        # return newly built model
        return model
    
    def lr_scheduler(self):
        return self.learning_rate
    
    def fit(self, x_train, y_train, batch_size, epochs, learning_rate) -> None:
        self.learning_rate = learning_rate
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def evaluate(
        self, x_test, y_test, batch_size, verbose
    ) -> Tuple[int, float, float]:
        return self.model.evaluate(
            x_test, y_test, batch_size=batch_size, verbose=verbose
        )
    
    def get_weights(self) -> fl.common.Weights:
        # fetch latest weights
        weights = self.model.extract_weights(style="pytorch")
        # return by casting to flower weights format
        return cast(fl.common.Weights, weights)
    
    def set_weights(self, weights: fl.common.Weights) -> None:
        # set the weights back to model
        self.model.setup_weights(weights, style="pytorch")
    