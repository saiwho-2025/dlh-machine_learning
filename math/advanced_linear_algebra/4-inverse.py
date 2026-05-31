#!/usr/bin/env python3
"""this module calculates the cofactor matrix of a matrix"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if len(matrix) == 0:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0

    for j in range(len(matrix)):
        sub_matrix = [
            row[:j] + row[j + 1:]
            for row in matrix[1:]
        ]

        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)

    return det


def minor(matrix):
    """calculates the minor matrix of a matrix"""
    if (not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for i in range(len(matrix)):
        row = []

        for j in range(len(matrix)):
            sub_matrix = [
                current_row[:j] + current_row[j + 1:]
                for k, current_row in enumerate(matrix)
                if k != i
            ]

            row.append(determinant(sub_matrix))

        minor_matrix.append(row)

    return minor_matrix


def cofactor(matrix):
    """calculates the cofactor matrix of a matrix"""
    minor_matrix = minor(matrix)

    cofactor_matrix = []

    for i in range(len(matrix)):
        row = []

        for j in range(len(matrix)):
            row.append(((-1) ** (i + j)) * minor_matrix[i][j])

        cofactor_matrix.append(row)

    return cofactor_matrix


def adjugate(matrix):
    """return the adjugate matrix of a matrix"""
    cofactor_matrix = cofactor(matrix)

    adjugate_matrix = [list(row) for row in zip(*cofactor_matrix)]

    return adjugate_matrix


def inverse(matrix):
    """return the inverse matrix of a matrix"""
    if (not isinstance(matrix, list)
            or matrix == []
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det_matrix = determinant(matrix)

    if det_matrix == 0:
        return None

    adjugate_matrix = adjugate(matrix)

    inverse_matrix = []

    for row in adjugate_matrix:
        inverse_row = []

    for value in row:
        inverse_row.append(value / det_matrix)

    inverse_matrix.append(inverse_row)

    return inverse_matrix
