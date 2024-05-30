#!/usr/bin/env python3
"""
This module provides a function to calculate the cost of a neural network with L2 regularization.
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    Returns:
    float: The cost of the network accounting for L2 regularization.
    """
    
    l2_norm_sum = 0
    for i in range(1, L + 1):
        weight_key = 'W' + str(i)
        l2_norm_sum += np.sum(np.square(weights[weight_key]))
    l2_reg_term = (lambtha / (2 * m)) * l2_norm_sum
    l2_reg_cost = cost + l2_reg_term
    return l2_reg_cost
