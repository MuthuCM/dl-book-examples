# Example 15.5
# Chat with PDF Version 1

# Step 0: Import Libraries
!pip install gradio
!pip install pypdf
import gradio as gr
from pypdf import PdfReader

# Step 2: Load the Hugging Face model for question answering
from transformers import pipeline
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 3: Define a Function to extract text from PDF document
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Step 4: Define a Function to answer questions
def answer_question(uploaded_file, user_question):
    # Extract text from the PDF file
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Get the answer from the question
    answer = qa_model(question=user_question, context=pdf_text)['answer']

    return answer

# Step 5: Create a Gradio interface using the new components
iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Your Question")
    ],
    outputs="text",
    title="Chat with Your PDF",
    description="Upload a PDF document and ask questions about its content."
)

# Launch the Gradio app
iface.launch()
