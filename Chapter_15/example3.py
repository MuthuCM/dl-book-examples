# Example 15.3

# Code Generator
# test_input = "def factorial():"

# Step 0: Import Libraries
!pip install gradio transformers torch
import gradio as gr
import torch

# Step 1:  Load the model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

# Step 1A: Check if CUDA (GPU) is available, else run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Function to generate code using the model
def generate_code(text):
    try:
        # Tokenize input text and move to the correct device
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        # Generate code using the model
        generated_ids = model.generate(input_ids, max_length=128, num_return_sequences=1)

        # Decode the generated tokens back into text
        generated_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_output
    except Exception as e:
        return f"Error: {str(e)}"

# Step 3: Create Gradio interface
interface = gr.Interface(fn=generate_code,
                         inputs="text",
                         outputs="text",
                         title="Code Generation with Salesforce CodeGen",
                         description="Enter a prompt, and this model will generate Python code based on the input.",
                         theme="compact")

# Launch the interface
interface.launch()
