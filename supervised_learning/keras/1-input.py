#!/usr/bin/env python3
"""Builds a neural network using the Keras Functional API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds and returns a Keras neural network."""

    # Create the input layer.
    # nx is the number of features in each input example.
    inputs = K.Input(shape=(nx,))

    # Start with the input as the previous layer.
    x = inputs

    # Build each layer of the network.
    for i in range(len(layers)):

        # Add a fully connected (Dense) layer.
        # x is passed into the new layer, connecting the layers together.
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        # Add dropout only to hidden layers.
        # The output layer should not have dropout.
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # Create the final model by specifying its inputs and outputs.
    model = K.Model(inputs=inputs, outputs=x)

    return model
