# Example 17.2
# Audio-to_Text Conversion
# Prepare the data
!pip install transformers

!pip install datasets

# Step 1 : Prepare the input data
from datasets import load_dataset
ds = load_dataset("librispeech_asr", split="train.clean.100", streaming=True,trust_remote_code=True )
# ds = load_dataset("librispeech_asr", split="train.clean.100", streaming=True,)

sample = next(iter(ds))
array = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]

# Let us get the sound for first 5 seconds
array = array[ : sampling_rate * 5]

# Code to listen to the first audio sample in the 100-hour split
import IPython.display as ipd
ipd.Audio(data = array, rate = sampling_rate)

# Step 2: Do Speech-to-Text Conversion(Transcription) using Whisper Model
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model = "openai/whisper-small",max_new_tokens = 200,)
pipe(array)


