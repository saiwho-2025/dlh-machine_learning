#!/usr/bin/env python3
"""Calculates the likelihood of obtaining data given probabilities."""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data given
    various hypothetical probabilities of developing severe side effects.

    Args:
        x: number of patients who develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of probability values

    Returns:
        numpy.ndarray containing the likelihood of x and n for each P value
    """
    if not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int):
        raise ValueError("x must be an integer that is" \
        " greater than or equal to 0")
    if x < 0:
        raise ValueError("x must be an integer that is " \
        "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    if len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    factorial = np.math.factorial
    combination = factorial(n) / (factorial(x) * factorial(n - x))

    return combination * (P ** x) * ((1 - P) ** (n - x))
