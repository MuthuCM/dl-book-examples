# Example 16.3
# Face Mask Detection
!pip install gradio transformers Pillow

# Use a pipeline as a high-level helper
from transformers import pipeline
from PIL import Image
pipe = pipeline("image-classification", model="AkshatSurolia/ConvNeXt-FaceMask-Finetuned")

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("AkshatSurolia/ConvNeXt-FaceMask-Finetuned")
model = AutoModelForImageClassification.from_pretrained("AkshatSurolia/ConvNeXt-FaceMask-Finetuned")

# Define a function to process and classify the uploaded image
def classify_image(image):
    # Convert the image to a format that the model can handle
    img = Image.fromarray(image)

    # Use the pipeline to predict
    predictions = pipe(img)

    # Return the top prediction
    return {pred['label']: pred['score'] for pred in predictions}

import gradio as gr
# Create a Gradio interface
iface = gr.Interface(
    fn=classify_image,  # The function for prediction
    inputs=gr.Image(type="numpy"),  # Input as an image, Use gr.Image directly
    outputs="label",  # Output the label and score
    title="Face Mask Classification",
    description="Upload an image to classify whether a face is wearing a mask or not using ConvNeXt."
)

# Launch the interface
iface.launch()
