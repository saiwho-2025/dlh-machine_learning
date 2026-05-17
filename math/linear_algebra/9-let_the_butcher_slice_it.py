#!/usr/bin/env python3
"""Slice a matrix."""

matrix = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]
mat1 = matrix[1:3]
mat2 = [[3, 4], [9, 10], [15, 16], [21, 22]]
mat3 = [[10, 11, 12], [16, 17, 18], [22, 23, 24]]
print("The middle two rows of the matrix are:\n[[ 7 8 9 10 11 12]\n[13 14 15 16 17 18]]")
print("The middle two columns of the matrix are:\n[[ 3 4]\n[ 9 10]\n[15 16]\n[21 22]]")
print("The bottom-right, square, 3x3 matrix is:\n[[10 11 12]\n[16 17 18]\n[22 23 24]]")