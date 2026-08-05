#!/usr/bin/env python3
"""Perform agglomerative clustering."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Perform agglomerative clustering using Ward linkage.

    Args:
        X: Dataset with shape (n, d).
        dist: Maximum cophenetic distance.

    Returns:
        Cluster assignments with shape (n,).
    """
    # Calculate the hierarchical linkage matrix.
    linkage = scipy.cluster.hierarchy.linkage(
        X,
        method='ward'
    )

    # Assign each point to a cluster.
    clss = scipy.cluster.hierarchy.fcluster(
        linkage,
        t=dist,
        criterion='distance'
    )

    # Draw the color-coded dendrogram.
    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    # Display the dendrogram as Figure 1.
    plt.show()

    # Return labels used by the main file.
    return clss