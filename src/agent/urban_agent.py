import os

from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


from langchain.agents import Tool, initialize_agent
from tools.elevation_tool import analyze_elevation
from tools.roads_tool import analyze_roads


load_dotenv()

def DEM_agent(user_query: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    template = """
    You are a DEM map reading agent. Your task is to help in finding relevant regions in DEM map as per user query.
    User query is : {user_query}

    At the end tell what is the total useful area
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["user_query"]
    )
    tools_for_agent = [
        Tool(name="Elevation Analyzer",
             func=analyze_elevation,
             description="Find the relevant region in DEM from min_elevation to max_elevation"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True)
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(user_query = user_query)}
    )



    output_stuff= result["output"]
    return output_stuff
# # Define tools for the agent
# elevation_tool = Tool(
#     name="Elevation Analyzer",
#     func=lambda x: analyze_elevation("data/elevation.tif", "data/neighborhoods.shp"),
#     description="Analyzes and maps elevation data."
# )

# roads_tool = Tool(
#     name="Road Infrastructure Analyzer",
#     func=lambda x: analyze_roads("data/roads.shp"),
#     description="Analyzes road infrastructure data."
# )

# # Initialize the LLM (e.g., using OpenAI's API)
# llm = OpenAI(temperature=0.5)

# # Create and configure the agent
# agent = initialize_agent([elevation_tool, roads_tool], llm, agent_type="zero-shot-react-description")

# # Example query for testing
# response = agent.run("Map out areas suitable for construction based on elevation and show nearby roads.")
# print(response)
