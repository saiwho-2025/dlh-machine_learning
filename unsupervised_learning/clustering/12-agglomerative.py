#!/usr/bin/env python3
"""Perform agglomerative clustering."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Cluster a dataset using Ward linkage.

    Args:
        X: Dataset with shape (n, d).
        dist: Maximum cophenetic distance.

    Returns:
        Cluster assignments with shape (n,).
    """
    # Calculate the Ward linkage matrix.
    linkage = scipy.cluster.hierarchy.linkage(
        X,
        method='ward'
    )

    # Assign points using the distance threshold.
    clss = scipy.cluster.hierarchy.fcluster(
        linkage,
        t=dist,
        criterion='distance'
    )

    # Display the dendrogram.
    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    # Show Figure 1.
    plt.show()

    # Return labels used to create Figure 2.
    return clss
