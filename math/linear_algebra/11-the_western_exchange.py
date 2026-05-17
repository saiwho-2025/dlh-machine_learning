#!/usr/bin/env python3
"""exchange the rows of a matrix"""


def np_transpose(matrix):
    """Transpose a matrix."""
    return [list(row) for row in zip(*matrix)]
