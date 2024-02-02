#!/usr/bin/env python3

"""
Function: poly_integral(poly, C=0)

Calculates the integral of a polynomial.

"""

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Parameters:
        poly (list): A list of coefficients representing a polynomial. The index of the list represents the power of x that the coefficient belongs to.
                     Example: if f(x) = x^3 + 3x + 5, poly is equal to [5, 3, 0, 1].
        C (int, optional): An integer representing the integration constant. Default is 0.

    Returns:
        list: A new list of coefficients representing the integral of the polynomial. The returned list is as small as possible.

    Notes:
        - If a coefficient is a whole number, it should be represented as an integer.
        - If poly or C are not valid, return None.
    """
    # Check if poly is a valid list of coefficients
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None
    for i in range(len(poly)-1, 0, -1):
        integral.append(poly[i]/(i+1))

    integral.append(poly[0])
    integral.append(C)

    if len(poly) == 1 and poly[0] == 0:
        integral = [C]
    for i in range(len(integral)):
        if integral[i] % 1== 0:
            integral[i] = int(integral[i])

    return integral[::-1]
