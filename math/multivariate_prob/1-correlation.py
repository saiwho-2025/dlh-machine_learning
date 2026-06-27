#!/usr/bin/env python3
"""this module caculates the correlation matrix"""
import numpy as np


def correlation(C):
    """C is a ndarray of shape (d, d)
        d is the number of dimensions"""
    
    if not isinstance (C, np.ndarray):
        raise TypeError("C must be a numpy.ndaary")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    corr = C / np.outer(std, std)

    return corr
  