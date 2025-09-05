# Example 15.5B
# Chat with PDF Version 3

# Step 0: Import Libraries
import gradio as gr
from pypdf import PdfReader
import nltk
# Download NLTK resources
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Step 1: Load the Hugging Face models for question answering and summarization
from transformers import pipeline
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
summarization_model = pipeline("summarization")

# Step 2: Define a function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Step 3: Define a function to summarize text
def summarize_text(text):
    # Split the text into chunks for summarization if needed
    summaries = summarization_model(text, max_length=130, min_length=30, do_sample=False)
    return summaries[0]['summary_text']

# Step 4: Define a function to answer a question using the summarized text
def answer_question(uploaded_file, user_question):
    pdf_text = extract_text_from_pdf(uploaded_file)
    if not pdf_text.strip():
        return "No text found in the PDF. Please upload a valid PDF."

    # Summarize the extracted text
    summarized_text = summarize_text(pdf_text)

    # Answer the user's question using the summarized text
    answer = qa_model(question=user_question, context=summarized_text)['answer']
    return answer

# Step 6: Create a Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Your Question")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Chat with Your PDF",
    description="Upload a PDF document and ask questions about its content."
)

# Launch the Gradio app
iface.launch()
