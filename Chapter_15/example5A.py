# Example 15.5A
# Chat with PDF Version 2
# Example 4.13
# Chat with PDF Version 2

# Step 0: Import Libraries
import gradio as gr
from pypdf import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Step 1: Load the Hugging Face model for question answering
from transformers import pipeline
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 2 : Define a function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Step 3: Define a function to split text into smaller chunks (paragraphs or sentences)
def split_text_into_chunks(text, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_chunk_size:
            chunks.append(chunk)
            chunk = ""
        chunk += sentence + " "
    if chunk:
        chunks.append(chunk)
    return chunks

# Step 4: Define a function to retrieve the most relevant chunk for the question
def retrieve_relevant_chunk(chunks, question):
    best_chunk = ""
    highest_score = 0
    for chunk in chunks:
        result = qa_model(question=question, context=chunk)
        if result['score'] > highest_score:
            highest_score = result['score']
            best_chunk = chunk
    return best_chunk

# Step 5: Define a function to answer a question using the relevant chunk of text
def answer_question(uploaded_file, user_question):
    pdf_text = extract_text_from_pdf(uploaded_file)
    chunks = split_text_into_chunks(pdf_text)
    relevant_chunk = retrieve_relevant_chunk(chunks, user_question)
    answer = qa_model(question=user_question, context=relevant_chunk)['answer']
    return answer

# Step 6: Create a Gradio interface
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