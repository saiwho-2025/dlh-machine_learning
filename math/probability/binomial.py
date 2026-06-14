#!/usr/bin/env python3
"""this module represents a binomial distribution"""


class Binomial:
    """The classic Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a Binomial instance.

        Arguments:
            data (list): if provided, estimate n and p from the data
            n (int): number of Bernoulli trials
            p (float): probability of a "success"
        """

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # From:
            # mean = n * p
            # variance = n * p * (1 - p)
            # variance / mean = 1 - p
            p = 1 - (variance / mean)

            n = round(mean / p)

            # Recalculate p using the rounded n
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""

        k = int(k)

        if k < 0 or k > self.n:
            return 0

        fact_n = 1
        for i in range(1, self.n + 1):
            fact_n *= i

        fact_k = 1
        for i in range(1, k + 1):
            fact_k *= i

        fact_n_k = 1
        for i in range(1, self.n - k + 1):
            fact_n_k *= i

        combination = fact_n / (fact_k * fact_n_k)

        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """calculate the value of the CDF using pmf"""        k = int(k)

        k = int(k)

        if k < 0 or k > self.n:
            return 1

        cdf_sum = 0
        for i in range(1, k+1):
            cdf_sum += self.pmf(i)
        return cdf_sum