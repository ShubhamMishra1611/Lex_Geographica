import streamlit as st
import folium
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from geopy.distance import geodesic
from typing import Dict, Any, Tuple
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
from pathlib import Path

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
    print(f"{llm_response = }")
    result = get_elevation_artifact(llm_response)
    return result

def plot_suitable_areas(suitable_areas: np.ndarray, min_elevation: int, max_elevation: int):
    """Create and return a matplotlib figure of suitable areas"""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(suitable_areas, cmap='Greens')
    plt.colorbar(im, ax=ax, label='1 = Suitable, 0 = Not Suitable')
    ax.set_title(f'Suitable Areas ({min_elevation}m-{max_elevation}m elevation)')
    return fig


@tool(response_format="content_and_artifact")
def find_route(
    start_location: str,
    end_location: str,
    transport_mode: str = 'walk'
) -> Tuple[str, Dict[str, Any]]:
    """
    Find the shortest route between two locations.
    
    Args:
        start_location (str): Starting location name or address
        end_location (str): Destination location name or address
        transport_mode (str): Mode of transport ('walk', 'drive', 'bike'). Defaults to 'walk'
    
    Returns:
        Tuple[str, Dict[str, Any]]: Content message and route data
    """
    # Get coordinates
    print("INSIDE FIND ROUTE FUNCTION")
    geolocator = Nominatim(user_agent="my_route_finder")
    
    start_location_data = geolocator.geocode(start_location)
    end_location_data = geolocator.geocode(end_location)
    
    if not (start_location_data and end_location_data):
        return "Could not find one or both locations", None
    
    start_coords = (start_location_data.latitude, start_location_data.longitude)
    end_coords = (end_location_data.latitude, end_location_data.longitude)
    
    # Create bounding box
    north = max(start_coords[0], end_coords[0]) + 0.01
    south = min(start_coords[0], end_coords[0]) - 0.01
    east = max(start_coords[1], end_coords[1]) + 0.01
    west = min(start_coords[1], end_coords[1]) - 0.01
    
    # Download street network
    
    graph = ox.graph_from_bbox(north, south, east, west, network_type=transport_mode)
    
    # Get nearest nodes
    start_node = ox.nearest_nodes(graph, start_coords[1], start_coords[0])
    end_node = ox.nearest_nodes(graph, end_coords[1], end_coords[0])
    
    try:
        route = nx.shortest_path(graph, start_node, end_node, weight='length')
    except nx.NetworkXNoPath:
        return "No route found between the specified locations", None
    
    # Extract route coordinates
    route_coords = []
    for node in route:
        point = graph.nodes[node]
        route_coords.append([point['y'], point['x']])
    
    # Calculate total distance
    total_distance = sum(
        geodesic(route_coords[i], route_coords[i+1]).kilometers
        for i in range(len(route_coords)-1)
    )
    
    # Create route data dictionary
    route_data = {
        'start_coords': start_coords,
        'end_coords': end_coords,
        'route_coords': route_coords,
        'total_distance': total_distance,
        'start_location': start_location,
        'end_location': end_location,
        'transport_mode': transport_mode
    }
    
    return f"Route found! Distance: {total_distance:.2f} km", route_data

def get_route_artifact(llm_response):
    """Extract the artifact from LLM response by executing the tool call"""
    print("Extract the artifact from LLM response by executing the tool call")
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        tool_call = llm_response.tool_calls[0]
        tool_response = find_route.invoke(tool_call)
        return {
            'llm_response': llm_response,
            'content': tool_response.content,
            'artifact': tool_response.artifact,
            'tool_call': tool_call
        }
    return None

def display_route_map(route_data: Dict[str, Any]) -> None:
    """Create and display a Folium map with the route"""
    print("Create and display a Folium map with the route")
    if not route_data:
        st.error("No route data available")
        return
    
    center_lat = (route_data['start_coords'][0] + route_data['end_coords'][0]) / 2
    center_lon = (route_data['start_coords'][1] + route_data['end_coords'][1]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add markers
    folium.Marker(
        route_data['start_coords'],
        popup=f"Start: {route_data['start_location']}",
        icon=folium.Icon(color='green')
    ).add_to(m)
    
    folium.Marker(
        route_data['end_coords'],
        popup=f"End: {route_data['end_location']}",
        icon=folium.Icon(color='red')
    ).add_to(m)
    
    # Draw route
    folium.PolyLine(
        route_data['route_coords'],
        weight=4,
        color='blue',
        opacity=0.8
    ).add_to(m)
    
    # Save to temporary file and display in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        m.save(tmp.name)
        st.components.v1.html(open(tmp.name, 'r').read(), height=600)
    
    # Clean up temporary file
    Path(tmp.name).unlink()

def process_user_request(query: str):
    """Process user request and determine which tool to use"""
    llm_response = llm_with_tools.invoke(query)
    print(f"LLM Response: {llm_response}")
    
    # Check which tool was called
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        tool_name = llm_response.tool_calls[0]['name']
        
        if tool_name == 'analyze_elevation':
            return get_elevation_artifact(llm_response)
        elif tool_name == 'find_route':
            result = get_route_artifact(llm_response)
            if result and result['artifact']:
                display_route_map(result['artifact'])
            return result
    
    return None

# Streamlit app
st.set_page_config(page_title="Geo Analysis Tools", layout="wide")
st.title("Geographic Analysis Assistant")

# Initialize LLM with both tools
@st.cache_resource
def init_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=1,
    )
    return llm.bind_tools([analyze_elevation, find_route])

llm_with_tools = init_llm()

# Single query input for both tools
query = st.text_input(
    "Enter your query (e.g., 'Find a route from Central Park, New york to Times Square, New york' or 'Analyze elevation between 500 and 600 meters')",
    key="query"
)

if st.button("Process"):
    with st.spinner("Processing..."):
        result = process_user_request(query)
        
        if result:
            st.success(result['content'])
            
            # If it's an elevation analysis, show the plot
            if isinstance(result['artifact'], np.ndarray):
                tool_args = result['tool_call']['args']
                min_elevation = tool_args['min_elevation']
                max_elevation = tool_args['max_elevation']
                
                fig = plot_suitable_areas(
                    result['artifact'],
                    min_elevation=min_elevation,
                    max_elevation=max_elevation
                )
                st.pyplot(fig)
                
                total_suitable = np.sum(result['artifact'])
                total_area = result['artifact'].size
                percentage_suitable = (total_suitable / total_area) * 100
                st.write(f"Suitable area percentage: {percentage_suitable:.2f}%")
        else:
            st.error("No analysis results were generated. Please try a different query.")