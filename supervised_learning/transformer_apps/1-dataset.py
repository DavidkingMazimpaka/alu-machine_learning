#!/usr/bin/env python3
"""This is a sample dataset script."""
    

import tensorflow as tf
import tensorflow_datasets as tfds

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

# Example usage
if __name__ == "__main__":
    dataset = Dataset()
    tokenizer_pt, tokenizer_en = dataset.tokenize_dataset(dataset.data_train)

    # Print some information about the tokenizers
    print(f"Portuguese Vocabulary Size: {len(tokenizer_pt.word_index)}")
    print(f"English Vocabulary Size: {len(tokenizer_en.word_index)}")
