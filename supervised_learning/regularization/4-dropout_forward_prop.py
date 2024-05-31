#!/usr/bin/env python3
"""a function that calculates the forward propagation with dropout"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Calculating the forward propagation with dropout
    Args:
        X: numpy.ndarray with shape (nx, m) that contains the input data
        nx: the number of input features to the neuron
        m: the number of data points
        weights: dictionary of the weights and biases of the NN
        L: the number of layers
        keep_prob: probability that a node will be kept
    """
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        A_prev = cache["A" + str(i - 1)]
        A = np.matmul(W, A_prev) + b
        if i < L:
            A_prev = np.tanh(A)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A = A * D
            A = A / keep_prob
            cache["D" + str(i)] = D
        else:
            A = np.exp(A) / np.sum(np.exp(A), axis=0, keepdims=True)
            cache["A" + str(i)] = A
    return cache
