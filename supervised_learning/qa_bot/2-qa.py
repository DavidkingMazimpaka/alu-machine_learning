def answer_loop(reference):
    while True:
        user_input = input("Q: ")
        
        # Check for exit conditions
        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break
        
        # Check if the user's input is found in the reference
        if user_input.lower() in reference.lower():
            print("A:", reference)
        else:
            print("A: Sorry, I do not understand your question.")

# Example usage
if __name__ == "__main__":
    reference_text = """
    This is a reference text. It contains information about various topics. 
    You can ask questions related to this text.
    """
    answer_loop(reference_text)
