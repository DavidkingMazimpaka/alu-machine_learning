#!/usr/bin/env python3
"""Function that slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """Function"""
    import numpy as np
    slices = [slice(None)] * matrix.ndim

    # Update the slice objects for the specified axes
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)

    # Apply the slices to the matrix
    sliced_matrix = matrix[tuple(slices)]

    return sliced_matrix
