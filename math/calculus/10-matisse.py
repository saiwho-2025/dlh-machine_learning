def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly: A list of coefficients where each index represents
              the power of x for that coefficient.

    Returns:
        A new list of coefficients representing the derivative,
        [0] if the derivative is 0, or None if poly is invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coefficient in poly:
        if not isinstance(coefficient, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    derivative = []

    for power in range(1, len(poly)):
        derivative.append(poly[power] * power)

    return derivative
