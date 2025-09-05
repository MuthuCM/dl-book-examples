# Example 16.1
# Generation of an image based on a text prompt
!pip install  -q diffusers transformers scipy torch

# Step 1: Load the Stable Diffusion model from Hugging Face
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Step 2: Define your text prompt
prompt = "An Astronaut riding a Horse"
# prompt = "A Car running on a Road"

# Step 3: Generate an image
image = pipe(prompt).images[0]
# Save the image
image.save("generated_image.png")
# Display the image
image.show()
from google.colab import files
# Download the generated image
files.download('generated_image.png')
