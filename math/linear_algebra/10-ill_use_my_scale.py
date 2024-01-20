#!/usr/bin/env python3

""" calculating matrix shape"""


def np_shape(matrix):
    """Find the matrix shape"""
    import numpy as np
    shape = np.shape(matrix)
    return shape