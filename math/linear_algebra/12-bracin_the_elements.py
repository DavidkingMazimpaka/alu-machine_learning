#!/usr/bin/env python3

"""Performing addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """ do the magic of some syntaxy"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
