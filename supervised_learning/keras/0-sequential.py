#!/usr/bin/env python3
"""this model  builds a neural network with the keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras.

    Args:
        nx: Number of input features.
        layers: List containing the number of nodes in each layer.
        activations: List containing the activation function for each layer.
        lambtha: L2 regularization parameter.
        keep_prob: Probability of keeping a node during dropout.

    Returns:
        The Keras model.
    """

    # Sequential creates a model where layers are connected
    # in the order they are added.
    model = K.Sequential()

    # Add each Dense layer and its corresponding Dropout layer.
    for i in range(len(layers)):

        # Dense is a fully connected neural network layer.
        # Each layer gets its own number of nodes and activation function.
        model.add(
            K.layers.Dense(
                layers[i],
                activation=activations[i],

                # L2 regularization penalizes large weights,
                # helping to reduce overfitting.
                kernel_regularizer=K.regularizers.l2(lambtha)
            )
        )

        # Keras Dropout expects the probability of DROPPING a node.
        # The assignment gives us the probability of KEEPING a node,
        # so we convert it using: dropout rate = 1 - keep_prob.
        #
        # Example: keep_prob = 0.8 -> dropout rate = 0.2.
        model.add(K.layers.Dropout(1 - keep_prob))

    # Define the input shape without using the Input class.
    # None means the model can receive any number of training examples.
    model.build((None, nx))

    # Return the completed neural network.
    return model