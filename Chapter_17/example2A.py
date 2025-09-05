# Example 17.2A
# Audio-to_Text Conversion
# Transcribe a longer(1-minute) audio
# Prepare the data
!pip install transformers

!pip install datasets

! pip install genaibook

# Step 1: Specify input data
from genaibook.core import generate_long_audio
long_audio = generate_long_audio()

# Step 2: Define Audio Model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline("automatic-speech-recognition", model = "openai/whisper-small", device = device)

# Step 3: Generate Text from Audio
pipe(long_audio, generate_kwargs = {"task" : "transcribe"},chunk_length_s = 5,batch_size = 8, return_timestamps = True,)# Example 17.2A


