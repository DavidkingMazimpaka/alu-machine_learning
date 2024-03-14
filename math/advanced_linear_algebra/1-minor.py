#!/usr/bin/env python3
""" Advanced Linear Algebra"""


def determinant(mat):
    """ Calculates the determinant of a square matrix """
    if len(mat) == 1:
        return mat[0][0]
    if len(mat) == 2:
        determ = ((mat[0][0] * mat[1][1])
                  - (mat[0][1] * mat[1][0]))
        return determ

    determ = 0
    for i, j in enumerate(mat[0]):
        rows = [row for row in mat[1:]]
        temp = []
        for row in rows:
            a = []
            for c in range(len(mat)):
                if c != i:
                    a.append(row[c])
            temp.append(a)
        determ += j * (-1) ** i * determinant(temp)
    return determ


def minor(mat):
    """ Calculates the minor of a matrix """
    if not isinstance(mat, list) or mat == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in mat):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(mat) for row in mat):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(mat) == 1:
        return [[1]]

    mino = []
    for x in range(len(mat)):
        temp = []
        for y in range(len(mat[0])):
            s = []
            for row in (mat[:x] + mat[x + 1:]):
                s.append(row[:y] + row[y + 1:])
            temp.append(determinant(s))
        mino.append(temp)
    return mino
