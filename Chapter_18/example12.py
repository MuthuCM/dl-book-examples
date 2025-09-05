# Example 18.12
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
controlnet = ControlNetModel.from_pretrained(
                                              "diffusers/controlnet-depth-sdxl-1.0",  \
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

from controlnet_aux import MidasDetector

! pip install genaibook
from PIL import Image
from genaibook.core import load_image, SampleURL

# loads the MiDAS depth detector model
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
controlnet_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder = "sdxl_models", weight_name = "ip-adapter_sdxl.bin")

# We are applying the IP-Adapter only to the mid block, which is where it should be mapped to the style in SDXL
scale = {"up":{"block_0" : [0.0, 1.0, 0.0]}}
controlnet_pipeline.set_ip_adapter_scale(scale)

original_image = load_image(SampleURL.WomanSpeaking)
original_image = original_image.resize((1024, 1024))

style_image = load_image(SampleURL.Mamoeiro)
style_image = style_image.resize((1024,1024))

# Apply MiDAS depth detection
processed_image_midas = midas(original_image).resize((1024, 1024), Image.BICUBIC)

image = controlnet_pipeline("A masked super hero singing a song",image = processed_image_midas,ip_adapter_image = style_image,controlnet_conditionong_scale = 0.5,num_inference_steps = 30).images[0]

from genaibook.core import image_grid, load_image, SampleURL
image_grid([original_image, style_image, processed_image_midas, image], rows = 1, cols = 4)

