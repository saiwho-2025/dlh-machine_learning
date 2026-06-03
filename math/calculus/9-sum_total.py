#!/usr/bin/env python3
"""this module caculates the sum of numbers from 1 to n to their power of 2"""

def summation_i_squared(n):
    if (not isinstance(n, int) or n<1):
        return None

    summation_i_squared(n) = n * (n + 1) * (2 * n + 1) // 6
    return summation_i_squared(n)
