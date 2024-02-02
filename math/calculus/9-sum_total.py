#!/usr/bin/env python3
"""
calculates the sum of squared elements
"""


def summation_i_squared(n):
    """summation of i
    
    Keyword arguments:
    Return: return None
    """
    if n == 1:
        return 1
    if n < 1:
        return None
    else:
        answer = (n*(n+1)*(2*n+1))//6
        return answer
    

#print(summation_i_squared(5))
    