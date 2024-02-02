#!/usr/bin/env python3
"""
derivative of a polynomial """


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    deri_val = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)-1, 0, -1):
        deri_val.append(poly[i] * i)
    return deri_val[::-1]


#print(poly_derivative([0, 1, 2, 3, 4, 5]))