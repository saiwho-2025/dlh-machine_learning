#!/usr/bin/env python3
"""Forward propagation with dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout."""
    cache = {}

    A = X

    for layer in range(1, L + 1):
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]

        Z = np.matmul(W, A) + b

        if layer == L:
            # Softmax for the output layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh for hidden layers
            A = np.tanh(Z)

            # Dropout mask
            D = np.random.binomial(1, keep_prob, size=A.shape)

            # Inverted dropout
            A = A * D
            A = A / keep_prob

            cache["D{}".format(layer)] = D

        cache["A{}".format(layer)] = A

    return cache
