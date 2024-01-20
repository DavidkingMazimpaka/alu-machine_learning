#!/usr/bin/env python3
'''
 This program add two matrices
'''
def add_matrices2D(mat1, mat2):
    '''
    This function computes two matrices
    of same length and return the sums element wise
    '''
    if (len(mat1) != len(mat2)) or (len(mat1[0]) != len(mat2[0])):
        return None
    matrix = [[] for _ in range(len(mat1))]
    for row in range(len(mat1)):
        for col in range(len(mat1[row])):
            matrix[row].append(mat1[row][col]+mat2[row][col])
    return matrix
