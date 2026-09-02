#!/usr/bin/env python3
"""Makes predictions using a neural network."""


def predict(network, data, verbose=False):
    """Makes a prediction using the neural network."""
    return network.predict(data, verbose=verbose)
