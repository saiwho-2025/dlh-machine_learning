"""Initialize centroids for K-means clustering."""

# Import NumPy for array operations and random number generation.
import numpy as np


# Define a function that accepts a dataset and number of clusters.
def initialize(X, k):
    """
    Initialize cluster centroids using a uniform distribution.

    Args:
        X: A NumPy array with shape (n, d).
        k: A positive integer representing the number of clusters.

    Returns:
        A NumPy array with shape (k, d), or None on failure.
    """
    # Confirm that X is a NumPy array with exactly two dimensions.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        # Return None because X is invalid.
        return None

    # Confirm that k is a positive integer and not a Boolean value.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        # Return None because k is invalid.
        return None

    # Store the number of data points and dimensions.
    n, d = X.shape

    # Confirm that the dataset contains points and dimensions.
    if n == 0 or d == 0:
        # Return None because the dataset is empty.
        return None

    # Find the minimum value of X in each dimension.
    minimums = np.min(X, axis=0)

    # Find the maximum value of X in each dimension.
    maximums = np.max(X, axis=0)

    # Generate all k centroids with one call to random.uniform.
    centroids = np.random.uniform(
        low=minimums,
        high=maximums,
        size=(k, d)
    )

    # Return the initialized centroids.
    return centroids
