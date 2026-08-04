#!/usr/bin/env python3
"""Calculate multivariate Gaussian probability densities."""

# Import NumPy to perform numerical array operations.
import numpy as np


def pdf(X, m, S):
    """
    Calculate multivariate Gaussian probability densities.

    Args:
        X: NumPy array with shape (n, d).
        m: NumPy array with shape (d,).
        S: NumPy array with shape (d, d).

    Returns:
        Density values, or None on failure.
    """
    # Validate X.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    # Validate m.
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    # Validate S.
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    # Read the number of points and dimensions.
    n, d = X.shape

    # Reject empty input.
    if n == 0 or d == 0:
        return None

    # Check the shape of the mean.
    if m.shape != (d,):
        return None

    # Check the shape of the covariance matrix.
    if S.shape != (d, d):
        return None

    # Calculate the covariance determinant.
    try:
        determinant = np.linalg.det(S)
    except (np.linalg.LinAlgError, TypeError):
        return None

    # Reject an invalid covariance determinant.
    if determinant <= 0 or not np.isfinite(determinant):
        return None

    # Calculate the covariance inverse.
    try:
        inverse = np.linalg.inv(S)
    except (np.linalg.LinAlgError, TypeError):
        return None

    # Calculate point-to-mean differences.
    differences = X - m

    # Calculate the squared Mahalanobis distances.
    mahalanobis = np.sum(
        (differences @ inverse) * differences,
        axis=1
    )

    # Calculate the normalization coefficient.
    coefficient = 1 / np.sqrt(
        ((2 * np.pi) ** d) * determinant
    )

    # Calculate all density values.
    P = coefficient * np.exp(-0.5 * mahalanobis)

    # Enforce the required minimum value.
    P = np.maximum(P, 1e-300)

    # Return an array with shape (n,).
    return P
