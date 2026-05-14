#!/usr/bin/env python3
"""Add 2 matrices by elements, which are addable, however the matrix shapes may differ."""


def add_matrices(matrix1, matrix2):
    """
    Add two matrices element-wise.
    
    Handles matrices of different shapes by adding only the overlapping elements.
    
    Args:
        matrix1: First matrix (list of lists)
        matrix2: Second matrix (list of lists)
    
    Returns:
        A new matrix with the same shape as the larger input matrix,
        where overlapping elements are summed.
    """
    rows = max(len(matrix1), len(matrix2))
    cols = max(len(matrix1[0]) if matrix1 else 0, len(matrix2[0]) if matrix2 else 0)
    
    result = []
    
    for i in range(rows):
        row = []
        for j in range(cols):
            val1 = matrix1[i][j] if i < len(matrix1) and j < len(matrix1[i]) else 0
            val2 = matrix2[i][j] if i < len(matrix2) and j < len(matrix2[i]) else 0
            row.append(val1 + val2)
        result.append(row)
    
    return result