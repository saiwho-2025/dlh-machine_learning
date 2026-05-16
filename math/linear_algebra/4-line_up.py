#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    # check if the lengths are the same
    if len(arr1) != len(arr2):
        return None

    # add the two arrays together element-wise and return the result
    return [a + b for a, b in zip(arr1, arr2)]
