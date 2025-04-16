import mysql.connector
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import MySQLTool

# Step 1: Create MySQL Database Connection
def create_mysql_connection():
    connection = mysql.connector.connect(
        host="127.0.0.1", 
        port=3306,# e.g. "localhost" or the IP address of your MySQL server
        user="root",    # Your MySQL username
        password="your_password",  # Your MySQL password
        database="chinook"   # The name of your database
    )
    return connection

# Step 2: Define the MySQL Tool for LangChain
def mysql_query_tool(query):
    connection = create_mysql_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

# Step 3: Integrating MySQL with LangChain
# We will create a custom tool using the MySQL querying function
mysql_tool = Tool(
    name="MySQL Database",
    func=mysql_query_tool,
    description="A tool for querying the MySQL database"
)

# Step 4: Define the Prompt Template
prompt_template = """
You are a helpful assistant. Given the following query, you should fetch the data from the MySQL database:

{query}

Return the result in a readable format.
"""

# Initialize OpenAI model (you can use other models if needed)
llm = OpenAI(temperature=0.7)

# Create a LangChain Agent with the MySQL Tool
tools = [mysql_tool]
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Step 5: Example Interaction
def agent_interaction(query):
    prompt = prompt_template.format(query=query)
    response = agent.run(prompt)
    return response

# Test with a sample query
query = "how many rows are in the albums table?"
response = agent_interaction(query)
print(response)
