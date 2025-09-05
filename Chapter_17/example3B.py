# Example 17.3B
# Text-to_Speech Conversion using MMS_TTS
!pip install transformers torch

import torch
# Step 1: Specify MMS-TTS Model
from transformers import VitsModel, VitsTokenizer, set_seed
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

# Specify Input
inputs = tokenizer(text = "Hello - my dog is cute", return_tensors = "pt")
set_seed(555)

# Step 3: Generate Output
with torch.inference_mode():
   outputs = model(inputs["input_ids"])

import IPython.display as ipd
# Access the waveform and sampling rate from the correct keys
ipd.Audio(data=outputs.waveform[0].cpu().numpy(), rate=model.config.sampling_rate)
