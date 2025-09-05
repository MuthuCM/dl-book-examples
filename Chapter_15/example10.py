# Example 15.10
# Company Brochure Generation
# FINAL VERSION
# Step 1: Import Libraries
import os
import requests
import json
from typing import List
from bs4 import BeautifulSoup
from IPython.display import display, Markdown, update_display
from google.colab import userdata
from openai import OpenAI
openai_api_key = userdata.get('cm_muthu')
openai = OpenAI(api_key = openai_api_key)

# Step 2 : Define website Class
class website:
  url : str
  title : str
  body : str
  links : str
  text : str
  def __init__(self, url):
    self.url = url
    response = requests.get(url)
    self.body = response.content
    soup = BeautifulSoup(self.body, 'html.parser')
    self.title = soup.title.string if soup.title else "No title found"
    if soup.body:
      for irrelevant in soup.body(["script", "style", "img", "input"]):
        irrelevant.decompose()
      self.text = soup.body.get_text(separator = "\n", strip = True)
    else:
      self.text = "No text found"
    self.links = [link.get('href') for link in soup.find_all('a', href=True)]

  def  get_contents(self):
    return f"Webpage Title:\n{self.title}\n Webpage Contents:\n {self.text}\n\n"

#ed = website("https://edwarddonner.com")
#print( ed.get_contents() )

link_system_prompt = "You are provided with a list of links found on a web page."
link_system_prompt += "You have to decide which of the links would be most relevant"
link_system_prompt += "to include in a brochure about the company, such as links to an"
link_system_prompt += "About page, or a company page, or careers/Job pages. \n"
link_system_prompt += "You should respond in JSON in this example:"
link_system_prompt += """
                      {
                        "links" : [
                          {
                            "type" : "about page", "url" : "https://edwarddonner.com/about"
                          },
                          {
                            "type" : "careers page", "url" : "https://edwarddonner.com/careers"
                          }
                        ]
                      }
                      """
#print(link_system_prompt)

# Function to form User prompt
def get_links_user_prompt(website):
  user_prompt = f"Here is the list of links on the website of {website.url} -"
  user_prompt += "Please decide which of these are relevant web links for"
  user_prompt += "a brochure about the company, respond with the full https URL:"
  user_prompt += "Do not include Terms of Service, Privacy, email links. \n"
  user_prompt += "Links(some might be relative links): \n"
  user_prompt += "\n".join(website.links)
  return user_prompt
#print(get_links_user_prompt(ed))

# Function to get relevant links
def get_links(url):
  website1 = website(url)
  prompt = get_links_user_prompt(website1)
  messages = [ {"role" : "system", "content" : link_system_prompt}, {"role" : "user", "content" : prompt}]
  completion = openai.chat.completions.create(model = 'gpt-4o-mini' , messages = messages, response_format={"type":"json_object"})
  result = completion.choices[0].message.content
  return json.loads(result)

#get_links("https://anthropic.com")

# Writing utility function to assemble all details into another prompt
def get_all_details(url):
  result = "Landing Page: \n"
  website1 = website(url)
  result += website1.get_contents()
  result += "\n\nRelevant Links: \n"
  links = get_links(url)
  print("Found Links:", links)
  for link in links['links']:
    result += f"\n\n{link['type']} \n"
    result += website(link['url']).get_contents()

  return result

#print(get_all_details("https://anthropic.com"))

# Defining System Prompt
system_prompt = "You are an assistant that analyzes the contents of several relevant pages"
system_prompt += " from a company website and creates a short brochure(summary) about"
system_prompt += "the company for prospective customers,investors and recruits. Respond in markdown."
system_prompt += "Include details of company culture, customers and careers/jobs"
system_prompt += "if you find relevant links. \n"

# Function to form User Prompt
def get_brochure_user_prompt(company_name, url):
  user_prompt = f"You are looking at a company called {company_name}\n"
  user_prompt += f"Here are the contents of its landing page and other relevant pages;"
  user_prompt += "use this information to build a short term brochure of this company"
  user_prompt += get_all_details(url)
  user_prompt = user_prompt[ :20000]
  return user_prompt

#get_brochure_user_prompt("Anthropic", "https://anthropic.com")

# Writing Functions to make a Brochure
def create_brochure(company_name,url):
  response = openai.chat.completions.create(model = 'gpt-4o-mini', messages = [ {"role" : "system", "content" : system_prompt}, {"role" : "user", "content" : get_brochure_user_prompt(company_name,url)}])
  result = response.choices[0].message.content
  display(Markdown(result))
#create_brochure("Anthropic", "https://anthropic.com")
