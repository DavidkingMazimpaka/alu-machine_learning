#!/usr/bin/env python3
""" 2-shuffle_data.py """

import numpy as np


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way
    Args:
        X: numpy.ndarray of shape (d, nx) that contains the input data
        Y: numpy.ndarray of shape (1, nx) that contains the one-hot
        encoding of the labels
    Returns: the shuffled X and Y matrices
    """
    shuff = np.random.permutation(X.shape[0])
    return X[shuff], Y[shuff]
