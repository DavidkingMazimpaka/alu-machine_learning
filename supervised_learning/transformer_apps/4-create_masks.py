#!/usr/bin/env python3
""" creating masks """


import tensorflow as tf

def create_masks(inputs, target):
    """
    Creates masks for the transformer model.

    Parameters:
        inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len_in).
        target (tf.Tensor): Target tensor of shape (batch_size, seq_len_out).

    Returns:
        encoder_mask (tf.Tensor): Padding mask of shape (batch_size, 1, 1, seq_len_in).
        combined_mask (tf.Tensor): Combined mask of shape (batch_size, 1, seq_len_out, seq_len_out).
        decoder_mask (tf.Tensor): Padding mask of shape (batch_size, 1, 1, seq_len_in).
    """
    # Create encoder padding mask
    encoder_mask = tf.cast(tf.equal(inputs, 0), tf.float32)  # Assuming 0 is the padding token
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]  # Shape (batch_size, 1, 1, seq_len_in)

    # Create decoder padding mask
    decoder_padding_mask = tf.cast(tf.equal(target, 0), tf.float32)  # Assuming 0 is the padding token
    decoder_padding_mask = decoder_padding_mask[:, tf.newaxis, tf.newaxis, :]  # Shape (batch_size, 1, 1, seq_len_out)

    # Create look-ahead mask for the decoder
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)  # Shape (seq_len_out, seq_len_out)

    # Combine decoder padding mask and look-ahead mask
    combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)  # Shape (batch_size, 1, seq_len_out, seq_len_out)

    return encoder_mask, combined_mask, decoder_padding_mask

# Example usage
if __name__ == "__main__":
    batch_size = 2
    seq_len_in = 5
    seq_len_out = 6

    # Dummy input and target tensors
    inputs = tf.constant([[1, 2, 0, 0, 0], [1, 2, 3, 4, 0]], dtype=tf.int32)  # Example input sentences
    target = tf.constant([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 5, 0]], dtype=tf.int32)  # Example target sentences

    encoder_mask, combined_mask, decoder_mask = create_masks(inputs, target)

    print("Encoder Mask Shape:", encoder_mask.shape)
    print("Combined Mask Shape:", combined_mask.shape)
    print("Decoder Mask Shape:", decoder_mask.shape)
