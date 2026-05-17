#!/usr/bin/env python3
"""conacatenate 2 matrices using numpy.concatenate"""

import numpy as np

def np_cat(mat1, mat2, axis=0):
    """Return the concatenation of two numpy.ndarray along a specific axis."""
    return np.concatenate((mat1, mat2), axis=axis)
