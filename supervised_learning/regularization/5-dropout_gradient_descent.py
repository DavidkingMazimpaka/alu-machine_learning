#!/usr/bin/env python3
"""a function that calculates the gradient with dropout"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights and biases of a NN
    using gradient descent with dropout
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache["A" + str(l - 1)] if l > 1 else cache["A0"]
        W = weights["W" + str(l)]
        b = weights["b" + str(l)]
        D = cache["D" + str(l)]
        dA = np.dot(W.T, dZ)
        dA *= D  # Apply dropout mask

        if l > 1:
            dA /= keep_prob  # Scale the gradients to account for dropout during training

        dZ = dA * (1 - np.square(A_prev))  # Derivative of tanh activation
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        weights["W" + str(l)] = W - alpha * dW
        weights["b" + str(l)] = b - alpha * db
