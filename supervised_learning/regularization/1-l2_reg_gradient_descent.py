#!/usr/bin/env python3
"""L2 regularization gradient descent."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using
    gradient descent with L2 regularization."""
    m = Y.shape[1]

    dz = cache["A{}".format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]

        dw = np.matmul(dz, A_prev.T) / m
        dw += (lambtha / m) * weights["W{}".format(layer)]

        db = np.sum(dz, axis=1, keepdims=True) / m

        # Calculate dz for the previous layer BEFORE updating W
        if layer > 1:
            dz = np.matmul(
                weights["W{}".format(layer)].T,
                dz
            ) * (1 - np.square(A_prev))

        weights["W{}".format(layer)] -= alpha * dw
        weights["b{}".format(layer)] -= alpha * db
