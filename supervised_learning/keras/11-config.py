#!/usr/bin/env python3
"""Saves and loads a Keras model configuration."""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration to a JSON file."""
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """Loads a model from a JSON configuration file."""
    with open(filename, 'r') as f:
        config = f.read()

    return K.models.model_from_json(config)
