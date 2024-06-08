#!/usr/bin/env python3
""" 0-norm_constants.py """

import numpy as np


def normalization_constants(X):
    """
    Function that calculates the normalization (standardization)
    constants of a matrix
    Args:
        X: is the numpy.ndarray of shape (m, nx) to normalize
            m is the number of data points
            nx is the number of features
    Returns: the mean and standard deviation of each feature, respectively
    """
    return X.mean(axis=0), X.std(axis=0)
