#!/usr/bin/env python3
"""Slice a matrix."""
import numpy as np
matrix = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])
mat1 = matrix[1:3]  # Middle two rows
mat2 = matrix[:, 1:3]  # Middle two columns
mat3 = matrix[2:, 2:]  # Bottom-right 3x3 matrix