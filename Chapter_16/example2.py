# Example 16.2
# Text-to-Image Generation
# Install the Monster API Python Package

!pip install gradio monsterapi -q
import gradio as gr
from monsterapi import client
# Initialize the Monster API client with your API key

api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6Ijg1OGI0MTIwYzQ0YzZkODI5MjgzZDM0NjFlMWY0YWRhIiwiY3JlYXRlZF9hdCI6IjIwMjQtMTAtMDlUMTQ6MzU6MTcuNTYxMTkwIn0.koOHRCm9xJHZIExKjgSdhIpFL12uOhcApcJyS2DfBKU'  # Replace with your actual Monster API key
monster_client = client(api_key)

# Define function to generate image

def generate_image(prompt, style):
    model = 'txt2img'  # Replace with the desired model name
    input_data = {
        'prompt': f'{prompt}, {style}',  # Combine prompt and style
        'negprompt': 'deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, mutated hands, fingers',
        'samples': 1,
        'enhance': False,
        'optimize': False,
        'safe_filter': True,
        'steps': 50,
        'aspect_ratio': 'square',
        'guidance_scale': 5.5,
    }

    # Call Monster API to generate image
    result = monster_client.generate(model, input_data)

    # Return the generated image URL
    return result['output'][0]
# Create Gradio interface

with gr.Blocks() as ImageGenerator:
    gr.Markdown("## Text-to-Image Generator with Monster API")

    prompt_input = gr.Textbox(label="Enter your prompt", placeholder="e.g. a girl in red dress")
    style_input = gr.Dropdown(
        choices=["watercolor", "photorealistic", "no style", "enhance", "anime",
                 "photographic", "digital-art", "comic-book", "fantasy-art",
                 "analog-film", "neonpunk", "isometric", "lowpoly", "origami",
                 "line-art", "craft-clay", "cinematic", "3d-model", "pixel-art",
                 "texture", "futuristic", "realism"],
        label="Choose a style"
    )
    output_image = gr.Image(label="Generated Image")

    generate_btn = gr.Button("Generate Image")

    # Set the function to be called on button click
    generate_btn.click(fn=generate_image, inputs=[prompt_input, style_input], outputs=output_image)

# Launch the Gradio interface
ImageGenerator.launch()


