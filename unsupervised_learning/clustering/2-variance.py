#!/usr/bin/env python3
"""Calculate total intra-cluster variance."""

# Import NumPy to perform numerical array operations.
import numpy as np


# Define the variance calculation function.
def variance(X, C):
    """
    Calculate total intra-cluster variance.

    Args:
        X: NumPy array with shape (n, d).
        C: NumPy array with shape (k, d).

    Returns:
        Total variance, or None when an input is invalid.
    """
    # Check that X is a two-dimensional NumPy array.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        # Return None when X is invalid.
        return None

    # Check that C is a two-dimensional NumPy array.
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        # Return None when C is invalid.
        return None

    # Extract the shapes of X and C.
    n, d = X.shape
    k, centroid_d = C.shape

    # Check that both arrays contain data.
    if n == 0 or d == 0 or k == 0:
        # Return None when either array is empty.
        return None

    # Check that X and C have matching dimension counts.
    if d != centroid_d:
        # Return None when their dimensions do not match.
        return None

    # Calculate each point-to-centroid coordinate difference.
    differences = X[:, np.newaxis, :] - C[np.newaxis, :, :]

    # Calculate every squared Euclidean distance.
    squared_distances = np.sum(differences ** 2, axis=2)

    # Select the smallest squared distance of each point.
    closest_distances = np.min(squared_distances, axis=1)

    # Add all closest squared distances.
    var = np.sum(closest_distances)

    # Return the total intra-cluster variance.
    return var
