#!/usr/bin/env python3
"""shuffle the data points in two matrices the same way"""

import numpy as np


def shuffle_data(X, Y):
    """arguments:
        X: numpy.ndarray of shape (m, nx)
        Y: numpy.ndarray of shape (m, ny)

        Returns:
            shuffled X and Y
    """
    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
