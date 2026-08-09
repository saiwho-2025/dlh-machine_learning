#!/usr/bin/env python3
"""Defines a deep neural network for binary classification."""

import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Represents a deep neural network."""

    # Keep your existing methods here:
    # __init__, properties, forward_prop, cost,
    # evaluate, gradient_descent and train.

    def save(self, filename):
        """Save the deep neural network to a pickle file."""
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
