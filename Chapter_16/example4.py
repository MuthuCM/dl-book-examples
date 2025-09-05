# Example 16.4
# Object Detection in an Image
!pip install gradio transformers torch matplotlib
import gradio as gr
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Load the model and feature extractor
model_name = "facebook/detr-resnet-50"
feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Function to perform object detection and return an image with boxes
def detect_objects(image):
    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    outputs = model(**inputs)

    # Get the detected boxes and labels
    target_sizes = torch.tensor([image.size[::-1]])  # Model expects (height, width)
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    # Visualize the results
    plt.imshow(image)
    ax = plt.gca()

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > 0.9:  # Filter by confidence threshold
            box = box.detach().numpy()
            x, y, w, h = box
            rect = Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
            plt.text(x, y, label_text, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')

    # Save the image with bounding boxes
    plt.savefig("detected_image.png", bbox_inches='tight')
    plt.close()

    return Image.open("detected_image.png")

# Gradio interface
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Object Detection",
    description="Upload an image and the model will detect objects with bounding boxes.",
)

# Launch the app
interface.launch()

