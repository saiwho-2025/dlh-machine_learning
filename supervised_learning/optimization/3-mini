#!/usr/bin/env python3
"""mini-batches for gradient descent for a neural network training"""

import numpy as np


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches for neural network training.

    Args:
        X: np.ndarray(m, nx)
        Y: np.ndarray(m, ny)
        batch_size: Size of each mini-batch.

    Returns:
        list of tuples (X_batch, Y_batch)
    """
    shuffle_data = __import__('2-shuffle_data').shuffle_data

    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []

    m = X.shape[0]

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches