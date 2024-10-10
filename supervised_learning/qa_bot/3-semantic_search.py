from sentence_transformers import SentenceTransformer, util
import os

def semantic_search(corpus_path, sentence):
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Read the corpus from the given path
    documents = []
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    
    # Encode the documents and the input sentence
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(sentence_embedding, document_embeddings)[0]
    
    # Get the index of the most similar document
    most_similar_idx = similarities.argmax().item()
    
    return documents[most_similar_idx]

# Example usage
if __name__ == "__main__":
    corpus_path = 'path/to/your/corpus'  # Replace with your corpus path
    search_sentence = "What is the importance of renewable energy?"
    result = semantic_search(corpus_path, search_sentence)
    print("Most similar document:\n", result)