#!/usr/bin/env python3
"""Initialize variables used by a Gaussian Mixture Model."""

# Import NumPy to manage numerical arrays.
import numpy as np

# Load the required K-means function.
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize Gaussian Mixture Model variables.

    Args:
        X: NumPy array with shape (n, d).
        k: Positive integer representing the cluster count.

    Returns:
        Priors, means, and covariance matrices.
    """
    # Validate X.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Validate k.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        return None, None, None

    # Read the number of points and dimensions.
    n, d = X.shape

    # Reject an empty data set.
    if n == 0 or d == 0:
        return None, None, None

    # Initialize equal cluster priors.
    pi = np.full(shape=(k,), fill_value=1 / k)

    # Initialize the means through K-means.
    m, clss = kmeans(X, k)

    # Check whether K-means succeeded.
    if m is None or clss is None:
        return None, None, None

    # Create one identity covariance matrix per cluster.
    S = np.tile(np.identity(d), (k, 1, 1))

    # Return all initialized variables.
    return pi, m, S
