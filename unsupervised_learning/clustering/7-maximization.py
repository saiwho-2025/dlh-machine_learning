#!/usr/bin/env python3
"""Calculate the maximization step of a Gaussian mixture model."""

# Import NumPy to handle numerical arrays.
import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step of a GMM.

    Args:
        X: NumPy array with shape (n, d).
        g: NumPy array with shape (k, n).

    Returns:
        Updated priors, means, and covariance matrices.
    """
    # Validate X.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Validate g.
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    # Read the data dimensions.
    n, d = X.shape

    # Read the responsibility dimensions.
    k, g_n = g.shape

    # Reject empty input.
    if n == 0 or d == 0 or k == 0:
        return None, None, None

    # Check that g contains one column per data point.
    if g_n != n:
        return None, None, None

    # Check that g contains numerical values.
    if not np.issubdtype(g.dtype, np.number):
        return None, None, None

    # Check that all responsibilities are finite.
    if not np.all(np.isfinite(g)):
        return None, None, None

    # Check that all responsibilities are nonnegative.
    if np.any(g < 0):
        return None, None, None

    # Check that each responsibility column sums to one.
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    # Calculate each cluster's total responsibility.
    cluster_weights = np.sum(g, axis=1)

    # Reject clusters having zero total responsibility.
    if np.any(cluster_weights <= 0):
        return None, None, None

    # Calculate the updated cluster priors.
    pi = cluster_weights / n

    # Calculate the updated centroid means.
    m = (g @ X) / cluster_weights[:, np.newaxis]

    # Create the covariance output array.
    S = np.zeros((k, d, d))

    # Update every covariance matrix.
    for cluster in range(k):
        # Calculate point-to-mean differences.
        differences = X - m[cluster]

        # Apply the current responsibilities.
        weighted_differences = (
            differences * g[cluster, :, np.newaxis]
        )

        # Calculate the weighted covariance matrix.
        S[cluster] = (
            weighted_differences.T
            @ differences
            / cluster_weights[cluster]
        )

    # Return the updated GMM variables.
    return pi, m, S