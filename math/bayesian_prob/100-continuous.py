#!/usr/bin/env python3
"""Calculates the Posterior probabilities of data."""

from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the likelihood of obtaining the data given
    various hypothetical probabilities of developing severe side effects.

    Args:
        x: number of patients who develop severe side effects
        n: to  tal number of patients observed
        P1: the lower bound on the range
        P2: the upper bound on the range

    Returns:
        the posterior probability of the intersection
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

    if not isinstance(p1, float) or p1 <= 0 or p1 > 1: 
        raise ValueError("p1 must be a float in the range [0, 1]") 

    if not isinstance(p2, float) or p2 <= 0 or p2 >1:
        raise ValueError("p2 must be a float in the range [0, 1]") 

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    a = x + 1
    b = n - x + 1

    return special.betainc(a, b, p2) - special.betainc(a, b, p1)
