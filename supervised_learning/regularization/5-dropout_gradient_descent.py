#!/usr/bin/env python3
"""Gradient descent with dropout."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights and biases using gradient descent with Dropout."""
    m = Y.shape[1]

    # Gradient of softmax + cross-entropy
    dz = cache["A{}".format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]

        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        # Calculate gradient for previous layer before updating weights
        if layer > 1:
            dz = np.matmul(weights["W{}".format(layer)].T, dz)

            # Tanh derivative
            dz = dz * (1 - np.square(A_prev))

            # Apply the same dropout mask and inverted dropout scaling
            dz = dz * cache["D{}".format(layer - 1)]
            dz = dz / keep_prob

        weights["W{}".format(layer)] -= alpha * dw
        weights["b{}".format(layer)] -= alpha * db
