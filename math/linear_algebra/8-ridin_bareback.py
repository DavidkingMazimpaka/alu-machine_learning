#!/usr/bin/env python3

"""Function for matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplies matrices"""
    if len(mat1[0]) != len(mat2):
        return None  # Matrices cannot be multiplied

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
