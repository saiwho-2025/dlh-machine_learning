#!/usr/bin/env python3
"""Saves and loads Keras models."""

import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire Keras model to a file."""

    # Save the complete model, including its architecture,
    # weights, optimizer configuration, and training configuration.
    network.save(filename)

    # The exercise requires this function to return None.
    return None


def load_model(filename):
    """Loads an entire Keras model from a file."""

    # Load and return the model stored in the specified file.
    return K.models.load_model(filename)
