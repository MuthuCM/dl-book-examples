# Example 15.8
# Conversational AI using Chat Interface
# Lesson 54, 55
# Step 1: Import Libraries
import os
from openai import OpenAI
from google.colab import userdata
import gradio as gr

# Step 2: Do Initialization
# openai_api_key = userdata.get("OPEN_API_key")
openai_api_key = userdata.get("cm_muthu")
openai = OpenAI(api_key = openai_api_key)
MODEL = "gpt-4o-mini"

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. Encourage the customer to buy hats if they are unsure what to get."

# Step 3: Define chat() function
def chat(message, history):
  messages = [{"role" : "system", "content" : system_message}]
  for user_message, assistant_message in history:
    messages.append({"role" : "user", "content" : user_message})
    messages.append({"role" : "assistant", "content" : assistant_message})
  messages.append({"role" : "user", "content" : message})
  if 'belt' in message:
    messages.append({"role" : "system", "content" : " The store does not sell belts, but be sure to point out that other items are on sale" })
  stream = openai.chat.completions.create(model = MODEL, messages = messages, stream = True)
  response = " "
  for chunk in stream:
    response += chunk.choices[0].delta.content or ''
    yield response

gr.ChatInterface(fn = chat).launch()


