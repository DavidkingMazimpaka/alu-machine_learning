#!/usr/bin/env python3
""" 0-convolve_grayscale_valid """


def convolve_grayscale_valid(images, kernel):
    """ performs a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = [[0] * output_w for _ in range(output_h)]

    for i in range(output_h):
        for j in range(output_w):
            for k in range(m):
                for x in range(kh):
                    for y in range(kw):
                        output[i][j] += images[k][i+x][j+y] * kernel[x][y]

    return output
