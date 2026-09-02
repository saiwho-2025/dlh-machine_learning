#!/usr/bin/env python3
"""Saves and loads the weights of a Keras model."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Saves the weights of a Keras model."""

    # Save only the model's learned weights.
    # save_format determines the format used to store the weights.
    network.save_weights(filename, save_format=save_format)

    # The function should return None.
    return None


def load_weights(network, filename):
    """Loads weights into a Keras model."""

    # Load the saved weights into the provided model.
    network.load_weights(filename)

    # The function should return None.
    return None
