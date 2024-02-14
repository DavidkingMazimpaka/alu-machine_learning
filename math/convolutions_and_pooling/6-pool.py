#!/usr/bin/env python3
""" convolutions with pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images
    """
    kh, kw = kernel_shape
    sh, sw = stride
    m, h, w, c = images.shape
    ch = int((h - kh) / sh) + 1
    cw = int((w - kw) / sw) + 1
    convoluted = np.zeros((m, ch, cw, c))
    for h in range(ch):
        for w in range(cw):
            square = images[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
            if mode == 'max':
                insert = np.max(square, axis=(1, 2))
            else:
                insert = np.average(square, axis=(1, 2))
            convoluted[:, h, w, :] = insert
    return convoluted
