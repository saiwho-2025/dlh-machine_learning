#!/usr/bin/env python3
"""Add two matrices."""


def add_matrices(mat1, mat2):
    """Add two matrices of the same shape."""
    if type(mat1) is not type(mat2):
        return None

    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None

        new_matrix = []
        for i in range(len(mat1)):
            result = add_matrices(mat1[i], mat2[i])
            if result is None:
                return None
            new_matrix.append(result)

        return new_matrix

    return mat1 + mat2
