#!/usr/bin/env python3
"""calculate the normalization constants of a matrix"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants
    of a matrix.

    X: numpy.ndarray of shape (m, nx)
       m = number of data points
       nx = number of features

    Returns:
        mean: numpy.ndarray of shape (nx,)
        std: numpy.ndarray of shape (nx,)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
