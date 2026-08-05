#!/usr/bin/env python3
"""Performs a t-SNE transformation."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """
    Perform a t-SNE transformation.

    Args:
        X: Dataset of shape (n, d).
        ndims: Dimensionality of the final representation.
        idims: Intermediate dimensionality after PCA.
        perplexity: Desired perplexity.
        iterations: Number of optimization iterations.
        lr: Learning rate.

    Returns:
        Y: Optimized representation of shape (n, ndims).
    """
    X_pca = pca(X, idims)
    P = P_affinities(X_pca, perplexity=perplexity)

    n = X.shape[0]
    Y = np.random.randn(n, ndims)
    update = np.zeros_like(Y)

    P *= 4

    for iteration in range(1, iterations + 1):
        momentum = 0.5 if iteration <= 20 else 0.8

        dY, Q = grads(Y, P)

        update = momentum * update - lr * dY
        Y += update
        Y -= np.mean(Y, axis=0)

        if iteration % 100 == 0:
            current_cost = cost(P, Q)
            print("Cost at iteration {}: {}".format(
                iteration, current_cost
            ))

        if iteration == 100:
            P /= 4

    return Y
