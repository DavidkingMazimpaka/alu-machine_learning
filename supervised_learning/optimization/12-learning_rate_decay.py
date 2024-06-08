#!/usr/bin/env python3
""" learning_rate_decay with tensorflow """

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ creates the learning rate using inverse time decay in tensorflow:
    Args: 
    alpha: is the original learning rate
    decay_rate: is the weight used to determine the rate at which alpha will decay
    global_step: is the number of passes of gradient descent that have elapsed
    decay_step: is the number of passes of GD
    the learning rate decay should occur in a stepwise fashion
    Returns: the learning rate decay operation"""
    return tf.train.inverse_time_decay(alpha, global_step, decay_step, 
                                       decay_rate, staircase_step=True)
