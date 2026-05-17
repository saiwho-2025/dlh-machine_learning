#!/usr/bin/env python3
"""Concatenate two matrices."""


def matrix_shape(matrix):
    """Return the shape of a matrix."""
    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape


def copy_matrix(matrix):
    """Return a deep copy of a matrix."""
    if not isinstance(matrix, list):
        return matrix

    return [copy_matrix(element) for element in matrix]


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis."""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    if len(shape1) != len(shape2) or axis >= len(shape1):
        return None

    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    if axis == 0:
        return copy_matrix(mat1) + copy_matrix(mat2)

    return [
        cat_matrices(mat1[i], mat2[i], axis - 1)
        for i in range(len(mat1))
    ]