import os
import atexit
import shutil
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Step 1: Define the local path to your pre-trained BlenderBot model and tokenizer
local_model_path = "chatbot/blenderbot"  # Update this to your actual local path

# Load the pre-trained BlenderBot model and tokenizer from the local directory
tokenizer = BlenderbotTokenizer.from_pretrained(local_model_path)
model = BlenderbotForConditionalGeneration.from_pretrained(local_model_path)

# Step 2: Define a function to interact with the chatbot
def interact_with_chatbot(user_input, conversation_history):
    # Step 2.1: Add user input to the conversation history
    conversation_history.append(f"User: {user_input}")
    
    # Step 2.2: Prepare the input text for the model
    conversation_text = " ".join(conversation_history[-5:])  # Use only the last 5 exchanges to keep context manageable
    
    # Step 2.3: Generate a response using the chatbot pipeline
    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True)
    response_ids = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    # Step 2.4: Add the generated response to the conversation history
    conversation_history.append(f"Chatbot: {response_text}")
    
    return response_text

# Step 3: Define a function to delete the model files from the cache directory
def delete_model_files():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--blenderbot-1B-distill")
    
    if os.path.exists(cache_dir):
        user_input = input("Do you want to delete the model files from the cache directory? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(cache_dir)
            print(f"Deleted model files from cache directory: {cache_dir}")
        else:
            print("Model files not deleted from cache directory.")
    else:
        print(f"Model files not found in cache directory: {cache_dir}")

# Step 4: Register the delete_model_files function to be called on program exit
atexit.register(delete_model_files)

# Step 5: Start the conversation loop
print("Welcome to the Indian Tourism Chatbot!")
print("Type 'quit' to end the conversation.\n")

conversation_history = []

while True:
    # Step 5.1: Get user input
    user_input = input("User: ")
    
    # Step 5.2: Check if the user wants to quit
    if user_input.lower() == 'quit':
        print("Thank you for using the Indian Tourism Chatbot. Goodbye!")
        break
    
    # Step 5.3: Generate and print the chatbot's response
    response = interact_with_chatbot(user_input, conversation_history)
    print(f"Chatbot: {response}")
