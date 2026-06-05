#!/usr/bin/env python3  # Tells the system to run this file using Python 3

"""This module calculates the integral of a polynomial."""  # Module description


def poly_integral(poly, C=0):  # Defines a function that integrates a polynomial
    """Calculate the integral of a polynomial."""  # Function description

    if not isinstance(poly, list) or not isinstance(C, int):  # Check that poly is a list and C is an integer
        return None  # Return None if poly or C is invalid

    if len(poly) == 0:  # Check if the polynomial list is empty
        return None  # Return None because an empty polynomial is invalid

    integral = [C]  # Start the result list with the integration constant

    for power, coefficient in enumerate(poly):  # Loop through each coefficient and its power
        if not isinstance(coefficient, (int, float)):  # Check that each coefficient is a number
            return None  # Return None if any coefficient is invalid

        new_coefficient = coefficient / (power + 1)  # Divide coefficient by the new power after integration

        if isinstance(new_coefficient, float) and new_coefficient.is_integer():  # Check if result is a whole number
            new_coefficient = int(new_coefficient)  # Convert whole-number floats to integers

        integral.append(new_coefficient)  # Add the new coefficient to the integral list

    while len(integral) > 1 and integral[-1] == 0:  # Remove unnecessary zeros from the end of the list
        integral.pop()  # Delete the last coefficient if it is zero

    return integral  # Return the final integral polynomial