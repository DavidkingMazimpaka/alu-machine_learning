#!/usr/bin/env python3
""" 1-convolve_grayscale_same """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Performs a same convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    output_h = h
    output_w = w
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel, 
                                     axis=(1, 2))

    return output
