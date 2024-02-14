#!/usr/bin/env python3
import numpy as np
""" 1-Convolve grayscale images """


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Grayscale images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images with the same dimensions as input images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for same convolution
    pad_h = kh // 2
    pad_w = kw // 2

    # Initialize output array
    output = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            # Extract the region of interest from the image
            roi = images[:, max(0, i - pad_h):min(h, i + pad_h + 1),
                         max(0, j - pad_w):min(w, j + pad_w + 1)]

            # Apply the kernel and sum the result
            output[:, i, j] = np.sum(roi * kernel, 
                                     axis=(1, 2))

    return output
