import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = hub.load('https://tfhub.dev/google/bert-uncased-tf2-qa/1')

def question_answer(question, reference):
    # Tokenize the inputs
    inputs = tokenizer.encode_plus(question, reference, add_special_tokens=True, return_tensors='tf')
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Get the model predictions
    outputs = model([input_ids, attention_mask])
    
    # Get the start and end logits
    start_logits = outputs['start_logits']
    end_logits = outputs['end_logits']
    
    # Convert logits to probabilities
    start_probs = tf.nn.softmax(start_logits, axis=-1).numpy()[0]
    end_probs = tf.nn.softmax(end_logits, axis=-1).numpy()[0]
    
    # Get the indices of the start and end of the answer
    start_index = np.argmax(start_probs)
    end_index = np.argmax(end_probs)
    
    # Check if the indices are valid
    if start_index <= end_index:
        # Convert the token indices back to the answer string
        answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
        answer = tokenizer.decode(answer_tokens)
        answer = answer.strip()
        
        # If the answer is empty, return None
        return answer if answer else None
    
    return None

# Example usage:
question = "What is the capital of France?"
reference = "The capital of France is Paris. It is known for its art, fashion, and culture."
answer = question_answer(question, reference)
print(answer)
