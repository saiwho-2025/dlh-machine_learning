#!/usr/bin/env python3
"""Module to add two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """adds two 2D matrices element-wise"""
    # Check if they have the same number of rows
    if len(mat1) != len(mat2):
        return None

    # Handle the edge case of empty matrices
    if len(mat1) == 0:
        return []

    # Check if the number of columns matches
    if len(mat1[0]) != len(mat2[0]):
        return None

    # Perform nested element-wise addition and return a new matrix
    return [
        [a + b for a, b in zip(row1, row2)]
        for row1, row2 in zip(mat1, mat2)
    ]
