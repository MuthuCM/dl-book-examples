# Example 18.10
# Image Variation
import torch
from diffusers import StableDiffusionXLPipeline

# Assuming you have a GPU, otherwise change to "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Initialize the pipeline 
sdxl_base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16",
)
sdxl_base_pipeline.to(device) 

# We load the IP-Adapter too
from PIL import Image
from genaibook.core import load_image, SampleURL
from genaibook.core import load_image # Importing the load_image function
sdxl_base_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder = "sdxl_models", weight_name = "ip-adapter_sdxl.bin")
# We can set the scale of how strong we want our IP-Adapter to impactour overall result
sdxl_base_pipeline.set_ip_adapter_scale(0.8)
image = load_image(SampleURL.ItemsVariation)
original_image = image.resize((1024, 1024))

# Create the image variation
generator = torch.Generator(device = device).manual_seed(1)
variation_image = sdxl_base_pipeline(prompt = " ", ip_adapter_image = original_image, num_inference_steps = 25, generator = generator,).images

# Make sure to import the necessary libraries
from genaibook.core import image_grid
from PIL import Image
image_grid([original_image, variation_image[0]], rows =1, cols = 2)
