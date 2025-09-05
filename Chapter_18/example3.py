# Example 18.3
# Image Editing with Semantic Guidance
# Code to generate an image of a photo of the face of a man
from diffusers import SemanticStableDiffusionPipeline
import requests
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Original Line - causes error on systems with low GPU memory.
#device = torch.device("cpu") #forcing the device to be CPU

# Increase the timeout for requests
# timeout is set to 60 seconds here, you can adjust it based on your network speed
timeout = 60

# Download the model with an increased timeout
semantic_pipeline = SemanticStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    variant="fp16",
    # Add the timeout argument here
    timeout=timeout
).to(device)

generator = torch.Generator(device=device).manual_seed(100) #Ensuring generator is on the same device as the model
out = semantic_pipeline(prompt = "a photo of the face of a man", negative_prompt = "low quality, deformed", generator = generator)

out.images[0]
