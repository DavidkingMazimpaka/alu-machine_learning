#!/usr/bin/env python3
'''
A function def cat_matrices(mat1, mat2, axis=0)
that concatenates two matrices along a specific axis
'''

def matrix_shape(matrix):
    """
    Get the shape of the matrix.

    """
    matrix_shape = []
    while type(matrix) is list:
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape

def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specific axis.
    """
    shape1, shape2 = matrix_shape(mat1), matrix_shape(mat2)

    # Check if matrices have the same number of dimensions
    if len(shape1) != len(shape2):
        return None
    
    # Check if matrices have the same size along non-concatenation axes
    if any(s1 != s2 for s1, s2 in zip(shape1, shape2) if s1 != shape1[axis] and s2 != shape2[axis]):
        return None

    return recursive_concat(mat1, mat2, axis, 0)

def recursive_concat(m1, m2, axis=0, current=0):
    """
    Recursively concatenate matrices along the specified axis.
    """
    if axis != current:
        return [recursive_concat(m1[i], m2[i], axis, current + 1) for i in range(len(m1))]
    
    m1.extend(m2)
    return m1
