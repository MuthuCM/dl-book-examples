# Example 16.1A
# Generating the image of a Tourist Spot
# Step 0
! pip install gradio
!pip install pydub

# Step 1
# Imports
import gradio as gr
from io import BytesIO
from PIL import Image
import base64
from pydub import AudioSegment
from pydub.playback import play
import IPython.display as ipd
from openai import OpenAI
from google.colab import userdata
openai_api_key = userdata.get('cm_muthu')
openai = OpenAI(api_key = openai_api_key)

# Step 2: Define a Function to generate an image of a Tourist Spot using DALL-E-3 Model
def artist(city):
   image_response = openai.images.generate(model = "dall-e-3",prompt = f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant style",size = "1024x1024",n = 1,response_format = "b64_json", )
   image_base64 = image_response.data[0].b64_json
   image_data = base64.b64decode(image_base64)
   return Image.open(BytesIO(image_data))

# Step 3 : Generate an image of a Tourist Spot using DALL-E-3 Model
image = artist("london")
display(image)

