#!/usr/bin/env python3
"""this module caculates the mean and covariance of set{X}"""
import numpy as np


def mean_cov(X):
    """
    Arg:
        X is a numpy.nd array of shape (n,d)
        n is the number of data points
        d is the number of dimensions in each data point
    """

    if not isinstance(X, np.ndarray) or X.ndim !=2:
        raise TypeError("X must be a 2D numpy.ndarray")
    
    n,d = X.shape

    if n < 2: 
        raise ValueError("X must contain multiple data points")
    
    mean = np.mean(X, axis = 0, keepdims = True)
    cov = np.matmul((X - mean).T, (X - mean))/(n-1)

    return mean, cov
