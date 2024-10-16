#!/usr/bin/env python3
"""
Calculates the symmetric P affinities of a dataset for t-SNE
"""

import numpy as np
from P_init import P_init  # Adjust the import based on your file structure
from entropy import HP  # Adjust the import based on your file structure


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset
    """
    n, _ = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        # Initialize binary search bounds for beta
        beta_low = None
        beta_high = None
        beta = 1.0  # Starting value for beta
        # Perform binary search for the correct beta value
        for _ in range(50):  # Limit iterations to avoid infinite loop
            Hi, Pi = HP(D[i, np.concatenate((np.arange(i), np.arange(i + 1, n)))], beta)
            H_diff = Hi - np.log2(perplexity)
            if abs(H_diff) <= tol:
                break  # Found suitable beta
            if H_diff > 0:  # H is greater than log(perplexity)
                beta_high = beta if beta_high is None else beta_high
                beta = beta / 2 if beta_low is None else (beta + beta_low) / 2
            else:  # H is less than log(perplexity)
                beta_low = beta if beta_low is None else beta_low
                beta = beta * 2 if beta_high is None else (beta + beta_high) / 2
        # Store the affinities in the P matrix
        P[i, np.concatenate((np.arange(i), np.arange(i + 1, n)))] = Pi
    # Make P symmetric
    P = (P + P.T) / 2
    # Normalize P to sum to 1 across rows
    row_sums = np.sum(P, axis=1, keepdims=True)
    # Check for zero row sums and handle them
    if np.any(row_sums == 0):
        print("Warning: Row sums are zero; setting P to zero.")
        P = np.zeros_like(P)  # If any row sum is zero, set P to zero
    else:
        P /= row_sums  # Normalize
    return P
