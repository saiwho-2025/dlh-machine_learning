#!/usr/bin/env python3
"""Slice a numpy.ndarray along specific axes."""


def np_slice(matrix, axes={}):
    """Slice matrix along given axes."""
    slices = [slice(None)] * matrix.ndim

    for axis, value in axes.items():
        slices[axis] = slice(*value)

    return matrix[tuple(slices)].copy()
