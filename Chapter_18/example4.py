# Example 18.4
# Image Editing with Semantic Guidance
# Code to make the man laugh
from diffusers import SemanticStableDiffusionPipeline
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
semantic_pipeline = SemanticStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype = torch.float16, variant = "fp16").to(device)
generator = torch.Generator(device = device).manual_seed(100)

out = semantic_pipeline(prompt = "a photo of the face of a man",
                        negative_prompt = "low quality, deformed",
                        editing_prompt = "smiling, smile",
                        edit_guidance_scale = 4,
                        edit_warmup_steps = 10,
                        edit_threshold = 0.99,
                        edit_momentum_scale = 0.3,
                        edit_mom_beta = 0.6,
                        reverse_editing_direction = False,
                        generator = generator)

out.images[0]

