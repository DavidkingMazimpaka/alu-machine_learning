#!/usr/bin/env python3
""" Module for advanced linear algebra operations. """
import numpy as np


def definiteness(matrix):
    """ Determines the definiteness of a matrix. """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    n = matrix.shape[0]
    if len(matrix.shape) != 2 or n != matrix.shape[1]:
        return None
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None
    eigenvalues, _ = np.linalg.eig(matrix)
    if all(eigenvalues > 0):
        return "Positive definite"
    if all(eigenvalues >= 0):
        return "Positive semi-definite"
    if all(eigenvalues < 0):
        return "Negative definite"
    if all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return 'Indefinite'
