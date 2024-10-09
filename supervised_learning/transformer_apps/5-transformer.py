#!/usr/bin/env python3
"""transformer"""


import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        
        self.wq = tf.keras.layers.Dense(dm)
        self.wk = tf.keras.layers.Dense(dm)
        self.wv = tf.keras.layers.Dense(dm)
        self.dense = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))
        return self.dense(output)

class Transformer(tf.keras.Model):
    def __init__(self, N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len_input, max_len_target):
        super(Transformer, self).__init__()
        self.encoder = ...  # Define your encoder
        self.decoder = ...  # Define your decoder
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        # Call encoder and decoder
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training, look_ahead_mask, decoder_mask)
        return self.final_layer(decoder_output)
