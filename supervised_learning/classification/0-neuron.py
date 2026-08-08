#!/usr/bin/env python3
"""defines a single neuron for binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron for binary classification."""


    def __init__(self, nx):
        """arg: nx is the number of input features to the neuron
            publice instance attributes:
            w is the weight vector for the neuron, 
            initialized using a random normal distribution
            b is the bias for the neuron, initialized to 0
            A is the activated output of the neuron(prediction), 
            initialized to 0"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
