#!/usr/bin/env python3
"""Z_norm to batch normalize the output"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    batch normalization an unactivated output of a neural network.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
