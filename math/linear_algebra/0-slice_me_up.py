#!/usr/bin/env python3

# Slicing a nested list matrix without modules
C = [list(range(i, i + 10)) for i in range(0, 50, 10)]
arr1 = C[0:1]
arr2 = C[0:4]
arr3 = [row[3:7] for row in C]
print(arr1, arr2, arr3, sep="\n")
