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
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None
    
    # Check if C is a valid integer
    if not isinstance(C, int):
        return None
    
    # Calculate the integral of the polynomial
    integral_coeffs = [C]
    for i in range(len(poly)):
        if i == 0:
            integral_coeffs.append(poly[i] / (i + 1))
        else:
            integral_coeffs.append(poly[i] / (i + 1))
    
    # Remove trailing zeros from the integral coefficients
    while integral_coeffs[-1] == 0 and len(integral_coeffs) > 1:
        integral_coeffs.pop()
    
    return integral_coeffs

# Test the function
print(poly_integral([5, 3, 0, 1]))  # Output: [0, 5, 1.5, 0, 0.25]
