from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from tools.elevation_tool import analyze_elevation
from tools.roads_tool import analyze_roads

# Define tools for the agent
elevation_tool = Tool(
    name="Elevation Analyzer",
    func=lambda x: analyze_elevation("data/elevation.tif", "data/neighborhoods.shp"),
    description="Analyzes and maps elevation data."
)

roads_tool = Tool(
    name="Road Infrastructure Analyzer",
    func=lambda x: analyze_roads("data/roads.shp"),
    description="Analyzes road infrastructure data."
)

# Initialize the LLM (e.g., using OpenAI's API)
llm = OpenAI(temperature=0.5)

# Create and configure the agent
agent = initialize_agent([elevation_tool, roads_tool], llm, agent_type="zero-shot-react-description")

# Example query for testing
response = agent.run("Map out areas suitable for construction based on elevation and show nearby roads.")
print(response)
