def main():
    while True:
        user_input = input("Q: ")
        
        # Check for exit conditions
        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break
        
        # Respond to the input
        print("A:", user_input)

if __name__ == "__main__":
    main()
