#!/usr/bin/env python3
""" Crops an image """


import tensorflow as tf


def crop_image(image, size):
    # Get the original image dimensions
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]
    
    # Extract the target height and width from the size tuple
    target_height, target_width = size
    
    # Ensure the crop size is not larger than the original image
    target_height = tf.minimum(target_height, original_height)
    target_width = tf.minimum(target_width, original_width)
    
    # Perform the random crop
    cropped_image = tf.image.random_crop(
        image, 
        size=[target_height, target_width, tf.shape(image)[2]]
    )
    
    return cropped_image
