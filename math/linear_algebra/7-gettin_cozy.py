#!/usr/bin/env python3
"""concatenate 2D matrices"""

def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate 2D matrices along a specified axis"""
    # Handle Row Concatenation (Stacking vertically)
    if axis == 0:
        # Check if the number of columns matches
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Simply combine the two lists of rows
        return mat1 + mat2

    # Handle Column Concatenation (Side-by-side)
    elif axis == 1:
        # Check if the number of rows matches
        if len(mat1) != len(mat2):
            return None
        # Join each row of mat1 with the corresponding row of mat2
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
