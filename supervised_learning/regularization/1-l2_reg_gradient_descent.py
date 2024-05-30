#!/usr/bin/env python3
"""a function that calculates the gradient of a neural network
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases using GD with L2 reg

    Parameters:
    Y (numpy.ndarray): matrix of shape (classes, m) with correct labels.
    weights (dict): Dictionary ofweights and biases.
    cache (dict): Dictionary of the outputs of each layer
    alpha (float): Learning rate.
    lambtha (float): L2 reg parameter.
    L (int): Number of layers in the network.
    Returns:
    None: The weights and biases are updated in place.
    """
    m = Y.shape[1]
    A_prev = cache["A" + str(L)]
    dZ = A_prev - Y

    for l in range(L, 0, -1):
        A_prev = cache["A" + str(l - 1)] if l > 1 else cache["A0"]
        W = weights["W" + str(l)]
        b = weights["b" + str(l)]

        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights["W" + str(l)] = W - alpha * dW
        weights["b" + str(l)] = b - alpha * db

        if l > 1:
            dA_prev = np.dot(W.T, dZ)
            dZ = dA_prev * (1 - np.square(A_prev))
