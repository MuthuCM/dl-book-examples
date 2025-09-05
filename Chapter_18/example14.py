# Example 18.14
# Image generation using LoRA fine-tuned Model
from diffusers import DiffusionPipeline
from huggingface_hub import model_info

# We will use a classic hand drawn cartoon style
lora_model_id = "alvdansen/littletinies"
# Determine the base model
# This information is frequently in the model card
# It is "stabilityai/stable-diffusion-xl-base-1.0" in this case
info = model_info(lora_model_id)
base_model_id = info.card_data.base_model

# Load the base model
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype = torch.float16)
pipe = pipe.to(device)

# Add the LoRA to the Model
pipe.load_lora_weights(lora_model_id)

# Add the LoRA to the Model
pipe.load_lora_weights(lora_model_id)

image = pipe("A llama drinking boba tea",
                      num_inference_steps = 25,
                      guidance_scale = 7.5).images[0]

display(image)



