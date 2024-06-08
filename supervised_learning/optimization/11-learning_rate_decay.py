#!/usr/bin/env python3
""" Learning rate decay"""


def learning_rate_decay(alpha, decay_rate, gloabl_step, decay_step):
    """ updates the learning rate using inverse time decay in numpy:
Args: 
    alpha: is the original learning rate
    decay_rate: is the weight used to determine the rate at which alpha will decay
    global_step: is the number of passes of gradient descent that have elapsed
    decay_step: is the number of passes of GD
    that should occur before alpha is decayed further
    the learning rate decay should occur in a stepwise fashion
Returns: 
    str:the updated value for alpha """
    return alpha / (1 + decay_rate * (gloabl_step / decay_step))
