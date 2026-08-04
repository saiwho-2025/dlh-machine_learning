#!/usr/bin/env python3
"""Calculate the expectation step of a Gaussian mixture model."""

# Import NumPy to handle numerical arrays.
import numpy as np

# Load the required Gaussian PDF function.
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step of a GMM.

    Args:
        X: NumPy array with shape (n, d).
        pi: NumPy array with shape (k,).
        m: NumPy array with shape (k, d).
        S: NumPy array with shape (k, d, d).

    Returns:
        Posterior probabilities and total log likelihood.
    """
    # Validate X.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # Validate pi.
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    # Validate m.
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    # Validate S.
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    # Read the data dimensions.
    n, d = X.shape

    # Read the cluster count.
    k = pi.shape[0]

    # Reject empty input.
    if n == 0 or d == 0 or k == 0:
        return None, None

    # Check the shape of the means.
    if m.shape != (k, d):
        return None, None

    # Check the shape of the covariance matrices.
    if S.shape != (k, d, d):
        return None, None

    # Check that the priors contain valid numbers.
    if not np.issubdtype(pi.dtype, np.number):
        return None, None

    # Check that all priors are finite and nonnegative.
    if not np.all(np.isfinite(pi)) or np.any(pi < 0):
        return None, None

    # Check that the priors sum to one.
    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Create the weighted density matrix.
    g = np.zeros((k, n))

    # Calculate each cluster's weighted densities.
    for cluster in range(k):
        # Evaluate all data points in the current distribution.
        density = pdf(X, m[cluster], S[cluster])

        # Check that the PDF calculation succeeded.
        if density is None:
            return None, None

        # Multiply each density by its cluster prior.
        g[cluster] = pi[cluster] * density

    # Add the weighted densities of all clusters.
    totals = np.sum(g, axis=0)

    # Reject invalid normalization values.
    if np.any(totals <= 0) or not np.all(np.isfinite(totals)):
        return None, None

    # Calculate the total log likelihood.
    l = np.sum(np.log(totals))

    # Normalize the weighted densities.
    g = g / totals

    # Return the posterior probabilities and log likelihood.
    return g, l
