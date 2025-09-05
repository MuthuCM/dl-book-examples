# Example 17.3
# Text-to_Speech Conversion using SpeechT5_TTS
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

!pip install genaibook

from genaibook.core import get_speaker_embeddings

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

inputs = processor(text = "There are llamas all around.", return_tensors = "pt")

speaker_embeddings = torch.tensor(get_speaker_embeddings()).unsqueeze(0)

import torch
with torch.inference_mode():
   spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
plt.figure()
plt.imshow(np.rot90(np.array(spectrogram)))
plt.show()

# Generate Speech(i.e., Audio)
from transformers import SpeechT5HifiGan
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
with torch.inference_mode():
   speech = vocoder(spectrogram)

# Code to listen to the generated audio
import IPython.display as ipd
ipd.Audio(speech.numpy(), rate=16000)

