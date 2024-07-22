#!/usr/bin/env python3
""" Flips an image horizontally """


import tensorflow as tf


def flip_image(image):
    # Using tf.image.flip_left_right to flip the image horizontally
    flipped_image = tf.image.flip_left_right(image)
    return flipped_image
