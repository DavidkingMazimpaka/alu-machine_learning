#!/usr/bin/env python3
'''
calculates the shape of a matrix
'''


def matrix_shape(matrix):
    '''
    this function computes the lengths and returns
    a shape
    '''
    rows = matrix
    shape = []
    while len(rows) > 0:
        shape.append(len(rows))
        rows = rows[0] if isinstance(rows[0], list) else []
    return shape
