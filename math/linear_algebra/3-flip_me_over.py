#!/usr/bin/env python3
"""This module defines a function to transpose a 2D matrix."""


def matrix_transpose(matrix):
    """Return a new transpose of the given 2D matrix."""
    return [list(row) for row in zip(*matrix)]
