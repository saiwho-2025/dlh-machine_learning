#!/usr/bin/env python3
"""Configures a neural network for optimization using Adam."""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization with categorical crossentropy loss."""

    # Create the Adam optimizer.
    # alpha is the learning rate.
    # beta_1 and beta_2 control how Adam updates the model's weights.
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    # Configure the model for training.
    # categorical_crossentropy is used for multi-class classification
    # when the target labels are one-hot encoded.
    # accuracy tells us how often the model predicts the correct class.
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # The exercise requires no return value.
    return None
