#!/usr/bin/env python3
""" This is a sample dataset2 script. """


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class Dataset:
    def __init__(self):
        # Load the TED translation dataset
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        
        # Initialize tokenizers
        self.tokenizer_pt = None
        self.tokenizer_en = None

        # Tokenize the datasets
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Parameters:
            data (tf.data.Dataset): A dataset of tuples (pt, en)

        Returns:
            tokenizer_pt: Tokenizer for Portuguese sentences
            tokenizer_en: Tokenizer for English sentences
        """
        # Extract Portuguese and English sentences from the dataset
        pt_sentences = []
        en_sentences = []

        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Create the tokenizers
        tokenizer_pt = tf.keras.preprocessing.text.Tokenizer(
            num_words=2**15, oov_token='<UNK>'
        )
        tokenizer_en = tf.keras.preprocessing.text.Tokenizer(
            num_words=2**15, oov_token='<UNK>'
        )

        # Fit the tokenizers on the respective sentences
        tokenizer_pt.fit_on_texts(pt_sentences)
        tokenizer_en.fit_on_texts(en_sentences)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        Parameters:
            pt (tf.Tensor): Tensor containing the Portuguese sentence
            en (tf.Tensor): Tensor containing the English sentence

        Returns:
            pt_tokens (np.ndarray): Array containing the Portuguese tokens
            en_tokens (np.ndarray): Array containing the English tokens
        """
        # Decode the tensors to strings
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Encode the sentences using the tokenizers
        pt_tokens = self.tokenizer_pt.texts_to_sequences([pt_sentence])[0]
        en_tokens = self.tokenizer_en.texts_to_sequences([en_sentence])[0]

        # Add start and end tokens
        start_token_pt = len(self.tokenizer_pt.word_index)  # Start token index
        end_token_pt = start_token_pt + 1  # End token index
        start_token_en = len(self.tokenizer_en.word_index)  # Start token index
        end_token_en = start_token_en + 1  # End token index

        pt_tokens = [start_token_pt] + pt_tokens + [end_token_pt]
        en_tokens = [start_token_en] + en_tokens + [end_token_en]

        return np.array(pt_tokens), np.array(en_tokens)

    @tf.function
    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method.

        Parameters:
            pt (tf.Tensor): Tensor containing the Portuguese sentence
            en (tf.Tensor): Tensor containing the English sentence

        Returns:
            pt_tokens (tf.Tensor): Tensor containing the Portuguese tokens
            en_tokens (tf.Tensor): Tensor containing the English tokens
        """
        pt_tokens, en_tokens = self.encode(pt, en)

        # Set shapes for the output tensors
        pt_tokens = tf.convert_to_tensor(pt_tokens, dtype=tf.int32)
        en_tokens = tf.convert_to_tensor(en_tokens, dtype=tf.int32)

        return pt_tokens, en_tokens

# Example usage
if __name__ == "__main__":
    dataset = Dataset()

    # Display shapes of tokenized examples
    for pt, en in dataset.data_train.take(1):
        print(f"Portuguese tokens: {pt.numpy()}")
        print(f"English tokens: {en.numpy()}")
