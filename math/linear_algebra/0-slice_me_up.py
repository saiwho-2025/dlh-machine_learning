#!/usr/bin/env python3
"""Module to slice a specific list into multiple sub-arrays"""


arr = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
arr1, arr2, arr3 = arr[:2], arr[-5:], arr[1:6]
print(arr1, arr2, arr3, sep="\n")
