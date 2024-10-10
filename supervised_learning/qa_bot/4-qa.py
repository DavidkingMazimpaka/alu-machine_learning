from sentence_transformers import SentenceTransformer, util
import os

def load_corpus(corpus_path):
    """Load all text documents from the specified directory."""
    documents = []
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def question_answer(corpus_path):
    """Answers questions based on multiple reference texts."""
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load the corpus
    documents = load_corpus(corpus_path)

    while True:
        question = input("Q: ")

        # Check for exit conditions
        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        # Encode the documents and the input question
        document_embeddings = model.encode(documents, convert_to_tensor=True)
        question_embedding = model.encode(question, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(question_embedding, document_embeddings)[0]

        # Get the index of the most similar document
        most_similar_idx = similarities.argmax().item()

        # Return the most relevant document as the answer
        print("A:", documents[most_similar_idx])

# Example usage
if __name__ == "__main__":
    corpus_path = 'path/to/your/corpus'  # Replace with your corpus path
    question_answer(corpus_path)
