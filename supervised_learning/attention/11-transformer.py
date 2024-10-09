#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create transformer network
"""


import tensorflow as tf

class SimpleEncoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_input, drop_rate=0.1):
        super(SimpleEncoder, self).__init__()
        # Define layers here (e.g., embedding, multi-head attention, feed forward)
        # For simplicity, we will just create a dense layer
        self.dense = tf.keras.layers.Dense(dm)

    def call(self, inputs, training, mask):
        # Implement the forward pass
        return self.dense(inputs)

class SimpleDecoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_target, drop_rate=0.1):
        super(SimpleDecoder, self).__init__()
        # Define layers here
        self.dense = tf.keras.layers.Dense(dm)

    def call(self, target, encoder_output, training, look_ahead_mask, decoder_mask):
        # Implement the forward pass
        return self.dense(target)

class Transformer(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = SimpleEncoder(N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = SimpleDecoder(N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training, look_ahead_mask, decoder_mask)
        final_output = self.linear(decoder_output)
        return final_output

# Example usage
if __name__ == "__main__":
    N = 6  # Number of layers
    dm = 512  # Dimensionality of the model
    h = 8  # Number of heads
    hidden = 2048  # Hidden size
    input_vocab = 10000  # Input vocabulary size
    target_vocab = 10000  # Target vocabulary size
    max_seq_input = 20  # Maximum input sequence length
    max_seq_target = 20  # Maximum target sequence length
    drop_rate = 0.1  # Dropout rate

    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate)

    # Dummy data for testing
    inputs = tf.random.uniform((32, max_seq_input, dm))  # Batch size of 32
    target = tf.random.uniform((32, max_seq_target, dm))  # Batch size of 32
    encoder_mask = None  # Placeholder for encoder mask
    look_ahead_mask = None  # Placeholder for look-ahead mask
    decoder_mask = None  # Placeholder for decoder mask

    output = transformer(inputs, target, training=True, encoder_mask=encoder_mask, look_ahead_mask=look_ahead_mask, decoder_mask=decoder_mask)
    print(f"Output shape: {output.shape}")  # Should be (32, max_seq_target, target_vocab)
