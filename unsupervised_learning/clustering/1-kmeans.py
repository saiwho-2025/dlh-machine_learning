#!/usr/bin/env python3
"""K-means clustering implementation."""

import numpy as np


# Define the K-means clustering function.
def kmeans(X, k, iterations=1000):
    """
    Apply K-means clustering to dataset X.

    Args:
        X: NumPy array with shape (n, d).
        k: Positive integer representing the cluster count.
        iterations: Positive integer limiting update steps.

    Returns:
        C and clss, or (None, None) when an input is invalid.
    """
    # Check that X is a two-dimensional NumPy array.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        # Return failure values when X is invalid.
        return None, None

    # Check that k is a positive integer and is not Boolean.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        # Return failure values when k is invalid.
        return None, None

    # Check that iterations is positive, integer, and not Boolean.
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations <= 0
    ):
        # Return failure values when iterations is invalid.
        return None, None

    # Extract the number of points and dimensions.
    n, d = X.shape

    # Check that the dataset is not empty.
    if n == 0 or d == 0:
        # Return failure values when X is empty.
        return None, None

    # Find the minimum value in each dimension.
    minimums = np.min(X, axis=0)

    # Find the maximum value in each dimension.
    maximums = np.max(X, axis=0)

    # Initialize all centroids with the first uniform call.
    C = np.random.uniform(
        low=minimums,
        high=maximums,
        size=(k, d)
    )

    # Repeat the assignment and update steps.
    for _ in range(iterations):
        # Save the current centroids to detect convergence.
        previous_C = C.copy()

        # Calculate point-to-centroid differences using broadcasting.
        differences = X[:, np.newaxis, :] - C[np.newaxis, :, :]

        # Calculate every point-to-centroid Euclidean distance.
        distances = np.linalg.norm(differences, axis=2)

        # Assign each point to its closest centroid.
        clss = np.argmin(distances, axis=1)

        # Update each centroid.
        for cluster in range(k):
            # Select points assigned to the current cluster.
            cluster_points = X[clss == cluster]

            # Check whether the current cluster has any points.
            if cluster_points.shape[0] > 0:
                # Replace the centroid with its assigned-point mean.
                C[cluster] = np.mean(cluster_points, axis=0)
            else:
                # Reinitialize an empty cluster with the second call.
                C[cluster] = np.random.uniform(
                    low=minimums,
                    high=maximums
                )

        # Stop when none of the centroids changed.
        if np.array_equal(C, previous_C):
            # Return the stable centroids and assignments.
            return C, clss

    # Recalculate distances using the final centroids.
    differences = X[:, np.newaxis, :] - C[np.newaxis, :, :]

    # Calculate the final Euclidean distances.
    distances = np.linalg.norm(differences, axis=2)

    # Produce assignments matching the final centroids.
    clss = np.argmin(distances, axis=1)

    # Return the final centroids and cluster assignments.
    return C, clss
