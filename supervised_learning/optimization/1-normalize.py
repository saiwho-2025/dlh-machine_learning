#!/usr/bin/env python3
"""the method to standarize a matrix"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    X: numpy.ndarray of shape (d, nx)
    m: numpy.ndarray of shape (nx,) - feature means
    s: numpy.ndarray of shape (nx,) - feature standard deviations

    Returns:
        The normalized X matrix.
    """
    return (X - m) / s
