#!/usr/bin/env python3
"""This module calculates the definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except Exception:
        return None

    tol = 1e-8

    if np.all(eigenvalues > tol):
        return "Positive definite"

    if np.all(eigenvalues >= -tol):
        return "Positive semi-definite"

    if np.all(eigenvalues < -tol):
        return "Negative definite"

    if np.all(eigenvalues <= tol):
        return "Negative semi-definite"

    if np.any(eigenvalues > tol) and np.any(eigenvalues < -tol):
        return "Indefinite"

    return None
