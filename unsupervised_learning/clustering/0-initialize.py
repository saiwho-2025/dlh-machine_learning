#!/usr/bin/env python3
"""this module initializes cluster centroids for K-means"""

import numpy as np
import matplotlib.pyplot as plt

def initialize(X, k):
    """
    Initialize centroids for K-means clustering.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Dataset containing n data points with d dimensions.
    k : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray of shape (k, d)
        Randomly initialized centroids, or None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None

    if type(k) is not int or k <= 0:
        return None

    n, d = X.shape

    if n == 0 or d == 0:
        return None

    minimums = np.min(X, axis=0)
    maximums = np.max(X, axis=0)

    centroids = np.random.uniform(
        low=minimums,
        high=maximums,
        size=(k, d)
    )

    return centroids
