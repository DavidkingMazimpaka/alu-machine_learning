#!/usr/bin/env python3
""" Adam class"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ Update variables using the Adam optimization algorithm
    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
        var (np.ndarray): variable to be updated
        grad (np.ndarray): gradient of var
        v (np.ndarray): the previous first moment of var
        s (np.ndarray): the previous second moment of var
        t (int): the time step
    Returns:
        np.ndarray: the updated variable and the new moment, respectively
    """
    v = beta1 * v + (1 - beta1) * grad
    v_corrected = v / (1 - (beta1 ** t))
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    s_corrected = s / (1 - (beta2 ** t))
    var = var - alpha * (v_corrected / ((s_corrected ** 0.5) + epsilon))
    return var, v, s
