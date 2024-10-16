#!/usr/bin/env python3
"""
Defines function that performs K-means on a dataset
"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    """
    # type checks to catch failure
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    # initialize cluster centroids using multivariate uniform distribution
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))
    # save copy of centroids to compare against later
    save_centroids = np.copy(C)
    if C.all() == saved_centroids.all():
        return C, clss
    saved_centroids = np.copy(C)
    return C, clss
