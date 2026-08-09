#!/usr/bin/env python3
"""a function that convert a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """one hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray):
        return None

    if one_hot.ndim != 2 or one_hot.shape[0] == 0:
        return None

    return np.argmax(one_hot, axis=0)
