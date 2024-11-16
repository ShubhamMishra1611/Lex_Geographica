import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

@tool(response_format="content_and_artifact")
def analyze_elevation(
    min_elevation: int,
    max_elevation: int,
) -> Tuple[str, np.ndarray]:
    """
    Analyze elevation data using DEM and return suitable areas.
    
    Args:
        min_elevation int: Minimum elevation threshold.
        max_elevation int: Maximum elevation threshold.
    Returns:
        Tuple[str, np.ndarray]: Content message and suitable areas array
    """
    dem_path = 'D:/LLM/Lexgis/Lex_Geographica/data/elevation.tif'
    dem_data = rasterio.open(dem_path)
    elevation = dem_data.read(1)
    
    suitable_areas = np.where(
        (elevation >= min_elevation) & (elevation <= max_elevation), 
        1, 
        0
    )
    
    return f"Analysis complete for elevation range {min_elevation}-{max_elevation}m", suitable_areas

def get_elevation_artifact(llm_response):
    """Extract the artifact from LLM response by executing the tool call"""
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        tool_call = llm_response.tool_calls[0]
        tool_response = analyze_elevation.invoke(tool_call)
        return {
            'llm_response': llm_response,
            'content': tool_response.content,
            'artifact': tool_response.artifact,
            'tool_call': tool_call  # Include the tool call for parameter access
        }
    return None

def process_elevation_request(query: str):
    """Process elevation analysis request and return both LLM response and artifact"""
    llm_response = llm_with_tools.invoke(query)
    result = get_elevation_artifact(llm_response)
    return result

def plot_suitable_areas(suitable_areas: np.ndarray, min_elevation: int, max_elevation: int):
    """Create and return a matplotlib figure of suitable areas"""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(suitable_areas, cmap='Greens')
    plt.colorbar(im, ax=ax, label='1 = Suitable, 0 = Not Suitable')
    ax.set_title(f'Suitable Areas ({min_elevation}m-{max_elevation}m elevation)')
    return fig

# Streamlit app
st.set_page_config(page_title="Elevation Analysis", layout="wide")
st.title("Elevation Analysis Tool")

# Initialize LLM
@st.cache_resource
def init_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=1,
    )
    return llm.bind_tools([analyze_elevation])

llm_with_tools = init_llm()

# Query input
query = st.text_input(
    "Enter your query (e.g., 'Analyze elevation between 500 and 600 meters')",
    key="query"
)

if st.button("Analyze"):
    with st.spinner("Processing..."):
        result = process_elevation_request(query)
        
        if result:
            st.success(result['content'])
            
            # Extract parameters from the tool call
            tool_args = result['tool_call']['args']
            min_elevation = tool_args['min_elevation']
            max_elevation = tool_args['max_elevation']
            
            # Get suitable areas and create plot
            suitable_areas = result['artifact']
            fig = plot_suitable_areas(
                suitable_areas,
                min_elevation=min_elevation,
                max_elevation=max_elevation
            )
            st.pyplot(fig)
            
            # Display basic statistics
            total_suitable = np.sum(suitable_areas)
            total_area = suitable_areas.size
            percentage_suitable = (total_suitable / total_area) * 100
            st.write(f"Suitable area percentage: {percentage_suitable:.2f}%")
        else:
            st.error("No analysis results were generated. Please try a different query.")