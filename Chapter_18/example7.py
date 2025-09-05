# Example 18.7
# Image Editing via Inversion
from genaibook.core import SampleURL, load_image, image_grid
from diffusers import LEditsPPPipelineStableDiffusion
import torch

# Increase the timeout duration for requests
timeout = 60  # Set timeout to 60 seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    # Add the timeout argument here
    timeout=timeout
)
pipe.to(device)

image = load_image(SampleURL.ManInGlasses).convert("RGB")
# Invert the image, gradually adding noise to it so
# it can be denoised with modified directions,
# effectively providing an edit
pipe.invert(image = image, num_inversion_steps = 50, skip = 0.2)

#Edit the image with an editing prompt
edited_image = pipe( editing_prompt ="glasses", \
                     # tell the model to remove the glasses by editing the direction  \
                     reverse_editing_direction = [True],  \
                     edit_guidance_scale =[1.5],  \
                     edit_threshold = [0.95], ).images[0]

image_grid([image, edited_image], rows = 1, cols = 2)



