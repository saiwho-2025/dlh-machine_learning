#!/usr/bin/env python3
"""Module for calculating the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix."""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i in range(len(matrix)):
        sub_matrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * determinant(sub_matrix)

    return det