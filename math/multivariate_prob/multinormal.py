#!/usr/bin/env python3
"""this module caculates the PDF of a class MultiNormal"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n)

        d = number of dimensions
        n = number of data points
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_centered = data - self.mean

        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        x is a numpy.ndarray of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean

        cov_inv = np.linalg.inv(self.cov) 
        cov_det = np.linalg.det(self.cov)

        exponent = -0.5 * ((diff.T @ cov_inv @ diff)[0, 0])

        denominator = np.sqrt(((2 * np.pi) ** d) * cov_det)

        pdf_value = np.exp(exponent) / denominator

        return pdf_value
