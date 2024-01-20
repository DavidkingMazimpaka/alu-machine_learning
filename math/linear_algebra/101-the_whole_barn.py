#!/usr/bin/env python3
'''
a program with a function def add_matrices(mat1, mat2): that adds two matrices:
'''
def add_matrices(mat1, mat2):
    '''
    a function def add_matrices(mat1, mat2): that adds two matrices:
    '''
    # Check if the matrices have the same shape
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if len(mat1) != len(mat2):
        return None

    # Initialize the result matrix with the same shape as mat1 and mat2
    result = []

    # Iterate through the rows of mat1 and mat2
    for i in range(len(mat1)):
        # Check if the rows have the same shape
        if len(mat1[i]) != len(mat2[i]):
            return None

        # Initialize the row of the result matrix
        row = []

        # Iterate through the elements of the rows and add them element-wise
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])

        # Add the row to the result matrix
        result.append(row)

    return result
