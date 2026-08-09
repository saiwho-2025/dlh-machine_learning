#!/usr/bin/env python3
"""Defines a deep neural network."""

import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Represents a deep neural network."""

    def __init__(self, nx, layers):
        # Existing constructor from 23-deep_neural_network.py
        pass

    # Keep all existing properties and methods:
    # forward_prop
    # cost
    # evaluate
    # gradient_descent
    # train

    def save(self, filename):
        """Save the network in pickle format."""
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Load a pickled deep neural network."""
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
