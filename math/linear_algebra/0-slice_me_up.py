#!/usr/bin/env python3

# Slicing a matrix using list comprehension for column extraction
C = [[j + i * 10 for j in range(10)] for i in range(5)]
arr1 = C[0:1]
arr2 = C[0:4]
arr3 = [row[3:7] for row in C]
print(arr1, arr2, arr3, sep="\n")
