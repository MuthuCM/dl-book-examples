# Example 18.1  
# Inpainting - Filling in missing parts of an image based on the surrounding context
! pip install genaibook
# Specify Model & Do Initialization
from genaibook.core import SampleURL, load_image, image_grid
from diffusers import StableDiffusionXLInpaintPipeline
import torch

inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype = torch.float16, variant = "fp16",).to(device)

img_url = SampleURL.DogBenchImage
mask_url = SampleURL.DogBenchMask
init_image = load_image(img_url).convert("RGB").resize((1014,1024))
mask_image = load_image(mask_url).convert("RGB").resize((1014,1024))

# Specify Input
prompt = "A majestic tiger sitting on a bench"

# Generate Output
# Ensure that the width and height are divisible by 8
width = (init_image.size[0] // 8) * 8  # Adjust width to be divisible by 8
height = (init_image.size[1] // 8) * 8 # Adjust height to be divisible by 8

image = inpaint_pipeline(prompt = prompt, image = init_image, mask_image = mask_image,
                        num_inference_steps = 50, strength = 0.80,
                        width = width, height = height).images[0]
image_grid([init_image, mask_image, image], rows = 1, cols = 3)