#!/usr/bin/env python3
""" Momentum Upgraded"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ Momentum """

    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
