#/usr/bin/env python3
""" Creating a Layer with Dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a tensorflow layer that includes dropout
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop = tf.layers.Dropout(keep_prob)
    model = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=drop)
    return model(prev)
