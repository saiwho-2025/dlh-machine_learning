#!/usr/bin/env python3
"""this module caculates the sum of numbers from 1 to n to their power of 2"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n.
    
    Args:
        n: the stopping condition(positive integer).
    
    Returns:
        Integer sum of squares if n is a valid positive integer, no None.
    """
    

    if type(n) is not int or n < 1:
        return None
    
    return n * (n + 1) * (2 * n + 1) // 6
