#!/usr/bin/env python3

"""
derivative of a polynomial 
"""


def poly_derivative(poly):
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        return [poly[i] * i for i in range(1, len(poly))] if sum(poly) != 0 else [0]


#print(poly_derivative([5, 3, 0, 1]))
