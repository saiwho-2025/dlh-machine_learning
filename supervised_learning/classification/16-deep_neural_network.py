#!/usr/bin/env python3
"""Defines a deep neural network for binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """Represents a deep neural network."""

    def __init__(self, nx, layers):
        """Initialize the deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for index in range(self.L):
            if type(layers[index]) is not int or layers[index] <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

            layer_number = index + 1

            if index == 0:
                previous_nodes = nx
            else:
                previous_nodes = layers[index - 1]

            self.weights["W{}".format(layer_number)] = (
                np.random.randn(layers[index], previous_nodes)
                * np.sqrt(2 / previous_nodes)
            )

            self.weights["b{}".format(layer_number)] = np.zeros(
                (layers[index], 1)
            )
