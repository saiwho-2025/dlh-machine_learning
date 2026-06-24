#!/usr/bin/env python3
"""Calculates the Posterior probabilities of data."""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the likelihood of obtaining the data given
    various hypothetical probabilities of developing severe side effects.

    Args:
        x: number of patients who develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of probability values
        Pr: 1D numpy.ndarray of prior beliefs of P

    Returns:
        numpy.ndarray containing the intersection values
    """
    if not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int):
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")
    if x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factorial = np.math.factorial
    combination = factorial(n) / (factorial(x) * factorial(n - x))
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    return likelihood * Pr


def marginal(x, n, P, Pr):
    """calculates the marginal probability of x based on intersection"""

    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """calculates the posterior probability in P"""

    posterior = intersection(x, n, P, Pr) / marginal(x, n, P, Pr)

    return posterior
