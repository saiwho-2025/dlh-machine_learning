#!/usr/bin/env python3
"""this module calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivatives of a polynomial"""
    if not isinstance(poly,int):
        return None
    
    if poly == 0:
        return [0]

    for power in range (len(coefficients)):
        poly_derivative(poly).coefficients[power]
    return poly_derivative(poly)