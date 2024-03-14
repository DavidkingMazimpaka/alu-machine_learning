#!/usr/bin/env python3
""" Module for advanced linear algebra operations. """


def adjugate(matrix):
    """ Calculates the adjugate of a matrix. """
    cofactors = minor(matrix)
    transposed = []
    for x in range(len(cofactors)):
        transposed.append([])
        for y in range(len(cofactors)):
            transposed[x].append(cofactors[y][x])
    return transposed


def determinant(matrix):
    """ Calculates the determinant of a square matrix. """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        determ = ((matrix[0][0] * matrix[1][1])
                  - (matrix[0][1] * matrix[1][0]))
        return determ

    determ = 0
    for i, j in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        temp = []
        for row in rows:
            sub_matrix = []
            for c in range(len(matrix)):
                if c != i:
                    sub_matrix.append(row[c])
            temp.append(sub_matrix)
        determ += j * (-1) ** i * determinant(temp)
    return determ


def minor(matrix):
    """ Calculates the minor of a matrix. """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minors = []
    for x in range(len(matrix)):
        row_minor = []
        for y in range(len(matrix[0])):
            sub_matrix = []
            for row in (matrix[:x] + matrix[x + 1:]):
                sub_matrix.append(row[:y] + row[y + 1:])
            sign = (-1) ** ((x + y) % 2)
            row_minor.append(determinant(sub_matrix) * sign)
        minors.append(row_minor)
    return minors
