#!/usr/bin/env python3
"""
Calculates the Shannon entropy and P affinities relative to a data point
"""


import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities for a data point

    Parameters:
        Di [numpy.ndarray of shape (n - 1,)]: pairwise distances 
            between a data point and all other points except itself
        beta [numpy.ndarray of shape (1,)]: beta value for the Gaussian distribution

    Returns:
        Hi [float]: the Shannon entropy of the points
        Pi [numpy.ndarray of shape (n - 1,)]: P affinities of the points
    """
    # Calculate the P affinities
    Pi = np.exp(-Di * beta[0])
    # Normalize P affinities to sum to 1
    Pi /= np.sum(Pi)
    # Calculate the Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi + 1e-12))  # Adding a small value to avoid log(0)
    return Hi, Pi
