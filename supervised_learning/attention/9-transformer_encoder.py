import tensorflow as tf
import numpy as np

# Assuming you have the positional encoding and EncoderBlock implemented correctly
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

class Encoder(tf.keras.layers.Layer):
    """
    Class to create an encoder for a transformer.
    
    Attributes:
        N (int): Number of blocks in the encoder.
        dm (int): Dimensionality of the model.
        embedding (tf.keras.layers.Embedding): Embedding layer for the inputs.
        positional_encoding (numpy.ndarray): Positional encodings.
        blocks (list): List of EncoderBlock instances.
        dropout (tf.keras.layers.Dropout): Dropout layer for positional encodings.
    """
    
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor.
        
        Parameters:
            N (int): Number of blocks in the encoder.
            dm (int): Dimensionality of the model.
            h (int): Number of heads.
            hidden (int): Number of hidden units in the fully connected layer.
            input_vocab (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length possible.
            drop_rate (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        
        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        
        # List of EncoderBlocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        
        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Calls the encoder and returns the output.
        
        Parameters:
            x (tensor): Input tensor of shape (batch, input_seq_len).
            training (bool): Determines if the model is in training mode.
            mask: Mask to be applied for multi-head attention.
        
        Returns:
            tensor: Encoder output of shape (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]
        
        # Embedding and positional encoding
        x = self.embedding(x)  # Shape: (batch, input_seq_len, dm)
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)
        
        # Pass through each EncoderBlock
        for block in self.blocks:
            x = block(x, training, mask)
        
        return x
