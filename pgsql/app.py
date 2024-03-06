from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.agents import create_sql_agent, create_json_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit, JsonToolkit
from langchain.sql_database import SQLDatabase
# import psycopg2
import os

load_dotenv()

# Initialize Ollama with the base URL and model
ollama_mistral = Ollama(base_url=os.getenv('OLLAMA_BASE_URL'), model="mistral")
ollama_llama2 = Ollama(base_url=os.getenv('OLLAMA_BASE_URL'), model="llama2")
ollama_sql = Ollama(base_url=os.getenv('OLLAMA_BASE_URL'), model="sqlcoder")

# Initialize SQLDatabase with the PostgreSQL URI
db = SQLDatabase.from_uri(os.getenv('POSTGRES_URI'))

# Define a function to prepare the agent prompt
def prepare_agent_prompt(input_text):
    agent_prompt = f"""
    Query the database using PostgreSQL syntax.
    Generate a PostgreSQL query using the input: {input_text}.
    Answer needs to be in the format of a JSON object.
    This object needs to have the key "query" with the SQL query and "query_response" as a MARKDOWN of the query response.
    """
    return agent_prompt

def prepare_json_agent_promt(input_text):
    agent_prompt = f"""
    Extract JSON from: {input_text}
    """
    return agent_prompt

user_prompt = input("Ask a question: ")
agent_prompt = prepare_agent_prompt(user_prompt)
table_list = db.get_usable_table_names()

# Create the SQL agent executor with the Ollama model and SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=ollama_mistral, table_list=table_list)

agent_executor = create_sql_agent(
    llm=ollama_mistral,
    toolkit=toolkit,
    verbose=True
)

spec = {
    "type": "object",
    "query": { "type": "string" },
    "query_response": { "type" : "string" },
    "required": ["query", "query_response"]
}

agent_json = create_json_agent(
    llm=ollama_mistral,
    toolkit=JsonToolkit(spec=spec),
    verbose=True
)

try:
    result = agent_executor.invoke(agent_prompt)
    result = agent_json.invoke(result)
    print(result)
    # print(f"Answer: {result['query']}")
except Exception as error:
    print(error)