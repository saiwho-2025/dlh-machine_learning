#!/usr/bin/env python3
"""Perform K-means clustering with Scikit-learn."""

# Import the only permitted module.
import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: Dataset with shape (n, d).
        k: Number of clusters.

    Returns:
        C: Centroids with shape (k, d).
        clss: Cluster assignments with shape (n,).
    """
    # Create the K-means model.
    model = sklearn.cluster.KMeans(n_clusters=k)

    # Train the model using X.
    model.fit(X)

    # Extract the centroid means.
    C = model.cluster_centers_

    # Extract each point's cluster index.
    clss = model.labels_

    # Return both results.
    return C, clss
