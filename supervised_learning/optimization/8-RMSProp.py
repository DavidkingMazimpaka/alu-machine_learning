#!/usr/bin/env python3
""" Training with momentum
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation for a neural network in tensorflow
        using the RMSProp optimization algorithm
        Args:
            loss: is the loss of the network
            alpha: is the learning rate
            beta2: is the RMSProp weight
            epsilon: is a small number to avoid division by zero
        Returns: the RMSProp optimization operation
    """
    train = tf.train.RMSPropOptimizer(
        alpha,
        decay=beta2,
        epsilon=epsilon
    )
    grads_and_vars = train.compute_gradients(loss)
    return train.apply_gradients(grads_and_vars)
