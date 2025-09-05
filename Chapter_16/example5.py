# Example 16.5
# Image Caption Generation
#Install the necessary libraries

!pip install transformers
!pip install gradio
force_download=True
# Build the Image Captioning Pipeline

from transformers import pipeline
image_captioner =pipeline("image-to-text",model="Salesforce/blip-image-captioning-large")
# Set up Prerequisites for Image Captioning App User Interface

import os
import io
import IPython.display
from PIL import Image
import base64

import gradio as gr

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))
def captioner(image):
    base64_image = image_to_base64_str(image)
    result = image_captioner(base64_image)
    return result[0]['generated_text']

gr.close_all()
# Build the Image Captioning App and Launch

ImageCaptionApp = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never")

ImageCaptionApp.launch()
