#!/usr/bin/env python3
"""Slice a matrix."""
matrix = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20],
          [21, 22, 23, 24, 25]]
mat1 = matrix[1:3]  # Middle two rows
mat2 = [matrix[0][1:3], matrix[1][1:3], matrix[2][1:3], matrix[3][1:3], matrix[4][1:3]]
mat3 = [matrix[2][2:], matrix[3][2:], matrix[4][2:]]
