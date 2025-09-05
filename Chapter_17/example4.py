# Example 17.4
# Multimodal AI Assistant(Text, Image, and Audio are generated)
# Lessons 62
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

# Step 1A: Define a function to get ticket price for destination city
ticket_prices = {"london" : "$799", "paris" : "$899", "tokyo" : "$1400", "berlin" : "$499"}
def get_ticket_price(destination_city):
  # print(f"Tool get_ticket_price called for {destination_city}")
  city = destination_city.lower()
  return ticket_prices.get(city, "Unknown")

# print(get_ticket_price("london"))

# Step 2: Define a function to generate an image
def artist(city):
   image_response = openai.images.generate(model = "dall-e-3",prompt = f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant style",size = "1024x1024",n = 1,response_format = "b64_json", )
   image_base64 = image_response.data[0].b64_json
   image_data = base64.b64decode(image_base64)
   return Image.open(BytesIO(image_data))
# image = artist("london")
# display(image)

# Step 3: Define a function to generate audio
def talker (city , price):
  message = f"Ticket price for {city} is {price}"
  response = openai.audio.speech.create(model = "tts-1", voice = "onyx", input = message)
  audio_stream = BytesIO(response.content)
  audio = AudioSegment.from_file(audio_stream, format = "mp3")
  audio.export("output.mp3", format = "mp3")
  play(audio)
# talker("london" , "$799")

# Step 4: Define generate_output() function
def generate_output(city_name):
   ticket_price = get_ticket_price(city_name)
   message = f"Ticket price for {city_name} is {ticket_price}"
   image = artist(city_name)
   talker(city_name , ticket_price)
   return message , image
# message, image = generate_output("london")
# display(image)

# Step 5: Create Gradio Interface
with gr.Blocks() as interface:
  destination_city = gr.Textbox(label = "Type Destination City name", placeholder ="e.g.London")
  ticket_price = gr.Textbox(label = "Ticket Price")
  destination_city_image = gr.Image(label = "Destination City Image")
  destination_city.submit(fn = generate_output, inputs = destination_city, outputs = [ticket_price, destination_city_image])

interface.launch()

