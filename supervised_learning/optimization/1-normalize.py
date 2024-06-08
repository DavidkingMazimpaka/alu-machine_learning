#!/usr/bin/env python3
""" 1-normalize.py """


def normalize(X, m, s):
    """ a function that normalizes (standardizes) a matrix
    Args: X: numpy.ndarray of shape (m, n) to normalize
          m: numpy.ndarray of shape (1, n) that contains the mean
          of X
          s: numpy.ndarray of shape (1, n) that contains the standard
          deviation of X
    """
    return (X - m) / s
