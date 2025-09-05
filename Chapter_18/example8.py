# Example 18.8 & 18.9
# Image Editing using ControlNet Model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
controlnet = ControlNetModel.from_pretrained(
                                              "diffusers/controlnet_depth_sdxl-1.0",  \
                                            torch_dtype = torch.float16, variant ="fp16", \
                                             ).to(device)

controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained( \
                               "stabilityai/stable-diffusion-xl-base-1.0", \
                               controlnet = controlnet,
                              torch_dtype = torch.float16, variant =  "fp16", \
                              ).to(device)

controlnet_pipeline.enable_model_cpu_offload()  # optional, saves VRAM
controlnet_pipeline.to(device)

!pip install controlnet_aux

# Image Editing using ControlNet Model after preprocessing the image
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from controlnet_aux import MidasDetector

! pip install genaibook

from PIL import Image
from genaibook.core import load_image, SampleURL
original_image = load_image(SampleURL.WomanSpeaking)
original_image = original_image.resize((1024, 1024))

# loads the MiDAS depth detector model
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

# Apply MiDAS depth detection
processed_image_midas = midas(original_image).resize((1024, 1024), Image.BICUBIC)

image = controlnet_pipeline( \
          "A colorful, ultra-realistic masked super hero singing a song",  \
          image = processed_image_midas,  \
          controlnet_conditionong_scale = 0.4,  \
          num_inference_steps = 30).images[0]

from genaibook.core import image_grid, load_image, SampleURL
image_grid([original_image, processed_image_midas, image], rows = 1, cols = 3)
