# Example 15.9
# Webpage Summarizer : Final Version
! pip install gradio
# Step 1: Import Libraries
from google.colab import userdata
from openai import OpenAI
openai_api_key = userdata.get('cm_muthu')
openai = OpenAI(api_key = openai_api_key)
import requests # Import the requests library here
from bs4 import BeautifulSoup

# Step 2 : Define website Class
class website:
  url : str
  title : str
  text : str
  def __init__(self, url):
    self.url = url
    response = requests.get(url)
    self.body = response.content
    soup = BeautifulSoup(self.body, 'html.parser')
    self.title = soup.title.string if soup.title else "No title found"
    for irrelevant in soup.body(["script", "style", "img", "input"]):
      irrelevant.decompose()
    self.text = soup.body.get_text(separator = "\n", strip = True)
  def  get_contents(self):
    return f"Webpage Title:\n{self.title}\n Webpage Contents:\n{self.text}\n\n"

#ws = website("https://anthropic.com")
#print( ws.get_contents() )

# Step 3: Specify System Prompt
system_prompt = "You are an assistant that analyzes the contents of a company website" 
system_prompt += "landing page and creates a short brochure(summary) about the company" 
system_prompt += "for prospective customers,investors and recruits. Respond in markdown."

# Step 4: Define stream_gpt() function 
def stream_gpt(prompt):
    messages = [ {"role" : "system", "content" : system_prompt}, {"role" : "user", "content" : prompt}]
    stream = openai.chat.completions.create(model = 'gpt-4o-mini' , messages = messages, stream = True)
    result = " "
    for chunk in stream:
        result += chunk.choices[0].delta.content or " "
    yield result

# Step 5: Define stream_brochure() function
def stream_brochure(company_name, url ):
    prompt = f"Please generate a company brochure for {company_name}."
    prompt += "Here is their landing page: \n"
    prompt += website(url).get_contents()
    result = stream_gpt(prompt)
    for chunk in result:
      yield chunk

# Step 6: Gradio Interface
import gradio as gr
view = gr.Interface(fn=stream_brochure,
                             inputs=[gr.Textbox(label="Company Name:"), gr.Textbox(label="Landing Page URL: ")],
                             outputs=gr.Markdown(label="Brochure:"), 
                             allow_flagging="never")
# Step 7: Launch Gradio Interface
view.launch()



