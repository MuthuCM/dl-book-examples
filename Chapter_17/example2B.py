# Example 17.2B
# Speech-to_Text Conversion(Transcription) using SpeechT5
# Prepare the data
!pip install transformers

!pip install datasets

from datasets import load_dataset
# ds = load_dataset("librispeech_asr", split="train.clean.100", streaming=True,trust_remote_code=True )
ds = load_dataset("librispeech_asr", split="train.clean.100", streaming=True,)

sample = next(iter(ds))

array = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]

from transformers import SpeechT5ForSpeechToText, SpeechT5Processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

import torch
inputs = processor(audio = array, sampling_rate = sampling_rate, return_tensors = "pt")
with torch.inference_mode():
   predicted_ids = model.generate(**inputs, max_new_tokens = 70)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True)
print(transcription)
