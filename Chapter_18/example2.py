# Example 18.2
# Prompt Weighting and Image Editing
from diffusers import DiffusionPipeline
import torch
import requests
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Changed to use CPU
#device = torch.device("cpu")
# Increase the timeout for requests
# timeout is set to 60 seconds here, you can adjust it based on your network speed
timeout = 60

# Load the model on the CPU instead of the GPU to avoid CUDA out of memory error
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", timeout=timeout).to(device)

! pip install compel

from compel import Compel, ReturnedEmbeddingsType
# Use the penultimate CLIP layer as it is more expressive
embeddings_type = (ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED)

# Instantiate Compel with text_encoder and tokenizer parameters
compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=embeddings_type,
                requires_pooled=[False, True])

! pip install genaibook

from genaibook.core import image_grid
# Prepare the Prompts
prompts = []
prompts.append("a humanoid robot eating pasta")
prompts.append("a humanoid++ robot eating pasta") # The + is equavalent to multiplying the prompt weight by 1.1
prompts.append('["a humanoid robot eating pasta", "a van goghpainting"].and(0.8, 0.2)')

# Making it van gogh!
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Original line - commented out
#device = torch.device("cpu")  # Force device to be CPU
images = []
for prompt in prompts:
  # Use the same seed across generations
  generator = torch.Generator(device=device).manual_seed(1) # Generator is now on CPU
  # The Compel library returns both conditioning vectors & pooled prompt embeds
  conditioning, pooled = compel(prompt)
  # We pass the conditioning embeds and pooled prompt embeds to the pipeline
  image = pipeline(prompt_embeds=conditioning,
                 pooled_prompt_embeds=pooled,
                 num_inference_steps=30,
                 generator=generator,).images[0]
  images.append(image)

image_grid(images, rows = 1, cols = 3)

  
