#!/usr/bin/env python3
"""This module prints the size of a square.

It imports the Square class, creates an instance, prints its size,
updates the size, and prints it again.
"""


def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
