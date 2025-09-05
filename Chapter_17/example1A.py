# Example 17.1A
# Audio Generation using TTS-1 Model
# Step 0

!pip install pydub

# Step 1
# Imports
#import gradio as gr
#from PIL import Image
#import base64
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import IPython.display as ipd
from openai import OpenAI
from google.colab import userdata
openai_api_key = userdata.get('cm_muthu')
openai = OpenAI(api_key = openai_api_key)

# Step 2: Define a function to generate audio
def talker (message):
  response = openai.audio.speech.create(model = "tts-1", voice = "alloy", input = message)
  audio_stream = BytesIO(response.content)
  audio = AudioSegment.from_file(audio_stream, format = "mp3")
  audio.export("audio_output.mp3", format = "mp3")
  play(audio)

# Step 3: Invoke talker() function
talker("Ticket Price is 110 dollars")