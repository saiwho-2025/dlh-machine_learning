#!/usr/bin/env python3
"""Find a suitable number of K-means clusters."""

# Import NumPy to validate array inputs.
import numpy as np

# Load the required K-means function.
kmeans = __import__('1-kmeans').kmeans

# Load the required variance function.
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test multiple cluster counts using total variance.

    Args:
        X: NumPy array with shape (n, d).
        kmin: Smallest cluster count.
        kmax: Largest cluster count.
        iterations: Maximum K-means update count.

    Returns:
        K-means results and variance differences.
    """
    # Validate X.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # Read the number of points and dimensions.
    n, d = X.shape

    # Reject an empty data set.
    if n == 0 or d == 0:
        return None, None

    # Validate kmin.
    if (
        not isinstance(kmin, int)
        or isinstance(kmin, bool)
        or kmin <= 0
    ):
        return None, None

    # Set an omitted kmax to the number of data points.
    if kmax is None:
        kmax = n

    # Validate kmax.
    if (
        not isinstance(kmax, int)
        or isinstance(kmax, bool)
        or kmax <= 0
    ):
        return None, None

    # Require at least two different cluster counts.
    if kmax <= kmin:
        return None, None

    # Validate iterations.
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations <= 0
    ):
        return None, None

    # Create the result lists.
    results = []
    d_vars = []

    # Prepare the baseline variance.
    base_var = None

    # Check each cluster count.
    for cluster_count in range(kmin, kmax + 1):
        # Run K-means.
        C, clss = kmeans(X, cluster_count, iterations)

        # Check the K-means output.
        if C is None or clss is None:
            return None, None

        # Calculate the current variance.
        current_var = variance(X, C)

        # Check the variance output.
        if current_var is None:
            return None, None

        # Save the current K-means output.
        results.append((C, clss))

        # Set the smallest cluster size as the baseline.
        if base_var is None:
            base_var = current_var

        # Save the variance reduction.
        d_vars.append(base_var - current_var)

    # Return both lists.
    return results, d_vars
