#!/usr/bin/env python3
"""the function calculates the minor of a matrix"""

def minor(matrix):
    """calculates the minor of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return [[1]]
    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
    minor_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            sub_matrix = [row[:j] + row[j + 1:] for row in matrix[:i] + matrix[i + 1:]]
            row.append(determinant(sub_matrix))
        minor_matrix.append(row)
    return minor_matrix