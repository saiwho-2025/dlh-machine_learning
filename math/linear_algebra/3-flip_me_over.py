#!/usr/bin/env python3

def matrix_transpose(matrix):
    """Return a new transpose of the given 2D matrix."""
    return [list(row) for row in zip(*matrix)]
