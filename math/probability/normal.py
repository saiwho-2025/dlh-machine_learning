#!/usr/bin/env python3
"""Using class Normal to represent an exponential distribution."""


class Normal:
    """The classic Normal distribution.
        data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the Exponential distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

        if len(data) < 2:
            raise ValueError("data must contain multiple values")

        self.mean = float(sum(data) / len(data))


