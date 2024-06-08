#!/usr/bin/env python3
""" 4-moving_average.py """

import numpy as np


def moving_average(data, beta):
    """ Function that calculates the weighted moving average of a data set
    Args:
    data: numpy.ndarray of shape (1, m) containing the data to calculate
    the moving average of
    beta: the weight used for the moving average
    Returns: a numpy.ndarray of shape (1, m) """
    v = 0
    result = []
    for x in range(len(data)):
        v = beta * v + (1 - beta) * data[x]
        b = 1 - (beta ** (x + 1))
        result.append(v / b)
    return result
