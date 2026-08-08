#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Represents a single neuron for binary classification."""

    def __init__(self, nx):
        """Initialize a neuron with nx input features."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weights vector."""
        return self.__W

    @property
    def b(self):
        """Return the bias."""
        return self.__b

    @property
    def A(self):
        """Return the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron."""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """Calculates the logistic regression cost"""
        m = Y.shape[1]

        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss)

        return cost
