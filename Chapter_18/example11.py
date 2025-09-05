# Example 18.11
# Image Prompting
import torch
from diffusers import StableDiffusionXLPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# We load the model and the IP-Adapter
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype = torch.float16).to(device)

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder = "sdxl_models", weight_name = "ip-adapter_sdxl.bin")

# We are applying the IP-Adapter only to the mid block, which is where it should be mapped to the style in SDXL
scale = {"up":{"block_0" : [0.0, 1.0, 0.0]}}
pipeline.set_ip_adapter_scale(scale)

! pip install genaibook
from PIL import Image
from genaibook.core import load_image, SampleURL
image = load_image(SampleURL.Mamoeiro)
original_image = image.resize((1024,1024))

display(original_image)

# Create the image variation
! pip install torch
import torch
generator = torch.Generator(device = device).manual_seed(1)
variation_image = pipeline(prompt = "a cat inside of a box", ip_adapter_image = original_image, num_inference_steps = 25, generator = generator,).images

# Make sure to import the necessary libraries
from genaibook.core import image_grid
from PIL import Image
image_grid([original_image, variation_image[0]], rows =1, cols = 2)

