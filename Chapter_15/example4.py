# Example 15.4
# A Basic Chatbot

# Step 0: Import Libraries
!pip install transformers torch gradio
# Importing the required libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

# Step 1: Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can change this to other models like gpt2-large
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 2: Function to refine the response into a more structured paragraph
def refine_response(response):
    # Strip leading/trailing whitespace
    refined = response.strip()

    # Ensure the first letter is capitalized (simple capitalization fix)
    if len(refined) > 0:
        refined = refined[0].upper() + refined[1:]

    # Add a period at the end if there's no punctuation
    if refined[-1] not in '.!?':
        refined += '.'

    return refined

# Step 3: Function to generate valid responses from the model
def generate_response(input_text, model, tokenizer, max_length=100, temperature=0.7, top_p=0.85):
    # Basic greeting handler for common inputs
    if input_text.lower() in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you today?"

    # Check if the input is empty
    if not input_text.strip():
        return "Please enter a message."

    # Encode the input and convert it to tensor
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response using the model
    outputs = model.generate(inputs,
                             max_length=max_length,
                             temperature=temperature,   # Adjust temperature for more coherent responses
                             top_p=top_p,               # Use top-p sampling to control randomness
                             top_k=50,                  # Control diversity with top-k
                             no_repeat_ngram_size=2     # Prevent repetition of phrases
                             )

    # Decode the generated tokens back into a string
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Refine the raw response into a cleaner paragraph
    return refine_response(raw_response)

# Step 4: Define the function that Gradio will use for the interface
def chatbot(user_input):
    # Generate a response from the model based on the user's input
    response = generate_response(user_input, model, tokenizer)
    return response

# Step 5: Create the Gradio interface
interface = gr.Interface(fn=chatbot,                         # The function to call for each input
                         inputs="text",                      # The input type (text box)
                         outputs="text",                     # The output type (text box)
                         title="Virtual Assistant",     # Title for the app
                         description="Ask anything and get responses.",  # Description
                         theme="compact")                    # A compact theme for the UI

# Launch the Gradio interface with an option to share the app publicly
interface.launch(share=True)

