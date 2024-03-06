import os
from dotenv import load_dotenv
# from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.llms import Ollama
from langchain_community.agent_toolkits import JsonToolkit

import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

# Load environment variables from a .env file
load_dotenv()

# Initialize Ollama with the base URL and model
llm = Ollama(base_url=os.getenv('OLLAMA_BASE_URL'), model="mistral")

# Get user input for the search query
user_prompt = input("Search for: ")

# Perform the DuckDuckGo search based on user input
ddg_search = DuckDuckGoSearchResults()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

prompt_template = "Summarize the following content: {content}"
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)

tools = [ddg_search, web_fetch_tool, summarize_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

prompt = "Research the Wood Plastic Compound Doors market. Use your tools to search and summarize content into a report on current state of the market."

print(agent.run(prompt))
