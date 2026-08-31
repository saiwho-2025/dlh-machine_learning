#!/usr/bin/env python3

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras.
    """

    # Create a sequential model: layers are connected in order.
    model = K.Sequential()

    # Add each Dense layer.
    for i in range(len(layers)):

        # Fully connected layer with the specified number of nodes
        # and activation function.
        model.add(
            K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            )
        )

        # Add dropout only to hidden layers.
        # The output layer should not have dropout.
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    # Define the input shape without using the Input class.
    model.build((None, nx))

    return model
