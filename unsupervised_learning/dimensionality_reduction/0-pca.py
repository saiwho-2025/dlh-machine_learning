#!/usr/bin/env python3
"""Performs Principal Component Analysis (PCA)."""

import numpy as np


def pca(X, var=0.95):
    """
    Calculate the PCA weight matrix for a zero-centered dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        var (float): Fraction of the variance to retain.

    Returns:
        numpy.ndarray: Weight matrix of shape (d, nd).
    """
    _, singular_values, vh = np.linalg.svd(X, full_matrices=False)

    variances = singular_values ** 2
    cumulative_variance = np.cumsum(variances) / np.sum(variances)
    nd = np.searchsorted(cumulative_variance, var) + 1

    return vh[:nd].T
