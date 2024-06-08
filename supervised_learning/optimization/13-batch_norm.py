#!/usr/bin/env python3
""" Batch  Normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network using batch normalization:
    Args:
        Z is a numpy.ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
        gamma is a numpy.ndarray of shape (1, n) containing the scales for Z
        beta is a numpy.ndarray of shape (1, n)  containing the offsets for Z
        epsilon is a small number used to avoid division by zero
    Returns:
        the normalized Z matrix """
    mean = Z.mean(axis=0)
    std = Z.var(axis=0)
    Z_norm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Z_tilda = gamma * Z_norm + beta
    return Z_tilda
