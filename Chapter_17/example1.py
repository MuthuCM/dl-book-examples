# Example 17.1
# Audio Generation
!pip install diffusers transformers scipy torch

# Step 1: Define Audio Generation Model
from transformers import pipeline
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline("text-to-audio", model = "facebook/musicgen-small",device=device)

# Step 2: Specify Input
data = pipe("electric rock solo, very intense")

#  Step 3: Write Code to generate audio & listen to it
import IPython.display as ipd
ipd.Audio(data["audio"][0], rate = data["sampling_rate"])
