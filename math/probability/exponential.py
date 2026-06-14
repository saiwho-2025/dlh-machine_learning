#!/usr/bin/env python3
"""Using class Exponential to represent an exponential distribution."""


class Exponential:
    """The classic Exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the Exponential distribution."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
            return

        if not isinstance(data, list):
            raise TypeError("data must be a list")

        if len(data) < 2:
            raise ValueError("data must contain multiple values")

        self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """Calculate the value of the PDF for a given time period."""
        if x < 0:
            return 0

        e = 2.7182818285

        return self.lambtha * (e ** (-self.lambtha * x))
    
    def cdf(self, k):
        """the cumulative probability of k in this poisson distribution"""
        if x < 0:
            return 0
        
        e = 2.7182818285
        return 1 - (e ** (-self.lambtha * x))
