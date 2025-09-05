# Example 18.13
# Fine-tuning Diffusion Models using LoRA technique
! pip install torch
import torch

from diffusers import StableDiffusionPipeline
model_id = "Supermaxman/hubble-diffusion-1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype = torch.float16,).to(device)

my_prompt = "Hubble reveals a cosmic dance of binary stars: In this stunning new image from "
my_prompt += "the Hubble Space Telescope, a pair of binary stars orbit each other in a "
my_prompt += "mesmerizing ballet of gravity and light. The interaction between these two "
my_prompt += "stellar partners causes them to shine brighter, offering astronomers crucial "
my_prompt += "insights into the mechanics of dual-star systems."

pipe(my_prompt).images[0]

