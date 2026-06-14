#!/usr/bin/env python3
"""using class Poisson to represent a poisson distribution """


import numpy


class Poisson:
    """The classic Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the Poisson Distribution
            Arguments:
                data is a list of the data for estimation of the distribution
                lambtha is expected number of occurrences in a given time frame
                k is the given number of successes.
        """
        if data is None:
            self.lambtha = lambtha
            self.__data = None
        else:
            self.data = data
            self.__lambtha = float(sum(self.__data) / len(self.__data))

    @property
    def data(self):
        """data property"""
        return self.__data

    @data.setter
    def data(self, value):
        if value is None:
            self.__data = None
            return
        if not isinstance(value, list):
            raise TypeError("data must be a list")
        if len(value) < 2:
            raise ValueError("data must contain multiple values")
        self.__data = value

    @property
    def lambtha(self):
        """lambtha property"""
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, value):
        if value <= 0:
            raise ValueError("lambtha must be a positive value")
        self.__lambtha = float(value)

    def pmf(self, k):
        """The value of the PMF for a given number of successes."""
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285
        factorial = 1

        for i in range(1, k + 1):
            factorial *= i

        return ((self.__lambtha ** k) * (e ** (-self.__lambtha))) / factorial

    def cdf(self,k):
        """the cumulative probability of k in this poisson distribution"""
        k = int(k)

        if k < 0:
            return 0
        
        cdf_value = 0

        for i in range(k+1):
            cdf_value += self.pmf(i)
        
        return cdf_value
        

    
