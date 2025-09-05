# Example 15.1  
# Text Generation
# Step 1 : Specify Model
from transformers import pipeline
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_generator = pipeline("text-generation", device = device)
# Step 2: Specify Input
prompt = "It was a dark and stormy"
# Step 3: Generate Output
text_generator(prompt, num_inference_steps = 10)[0][ "generated_text"]          