#!/usr/bin/env python3
"""
calculates the sum of squared elements
"""


def summation_i_squared(n):
    if not isinstance(n, int) or n < 1:
        return None
    elif n == 1:
        return 1
    else:
        return n**2 + summation_i_squared(n-1)
    

#print(summation_i_squared(5))
    