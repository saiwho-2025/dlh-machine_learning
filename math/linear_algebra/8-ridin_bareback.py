#!/usr/bin/env python3
"""matrix dot multiplication"""
def mat_mul(mat1, mat2):
    # Get dimensions
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])

    # Rule: number of columns of mat1 must equal rows of mat2
    if cols1 != rows2:
        return None

    # Create the result matrix (m x p) filled with zeros
    # result[i][j] will be the dot product of mat1's row i and mat2's column j
    result = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            # Calculate the dot product
            dot_product = sum(mat1[i][k] * mat2[k][j] for k in range(cols1))
            row.append(dot_product)
        result.append(row)

    return result
