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
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
from pathlib import Path

# Import urban suitability analysis logic
from populationthing import calculate_suitability

load_dotenv()

# Tool: Urban Suitability Analysis
@tool(response_format="content_and_artifact")
def urban_suitability_analysis(city_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Analyze urban suitability for housing development.
    """
    try:
        # Configure and load city data
        ox.config(use_cache=True, log_console=True)
        gdf_city = ox.geocode_to_gdf(city_name)
        graph = ox.graph_from_place(city_name, network_type='all')
        buildings = ox.geometries_from_place(city_name, tags={'building': True})

        # Calculate population density and other factors
        building_areas = buildings.area
        total_building_area = building_areas.sum()
        avg_building_area = building_areas.mean()
        city_area = gdf_city.iloc[0].geometry.area
        population_density = total_building_area / city_area
        street_lengths = ox.graph_to_gdfs(graph, nodes=False, edges=True)['length']
        total_street_length = street_lengths.sum()

        # Assess suitability
        suitability_score = calculate_suitability(
            population_density, avg_building_area, total_street_length, city_area
        )

        urban_data = {
            'city': city_name,
            'total_area': city_area,
            'population_density': population_density,
            'avg_building_area': avg_building_area,
            'total_building_area': total_building_area,
            'total_street_length': total_street_length,
            'suitability_score': suitability_score,
            'recommendation': 'Recommended' if suitability_score > 0.7 else 'Not Recommended'
        }
        summary = (
            f"Urban Suitability Analysis for {city_name}\n"
            f"Suitability Score: {suitability_score:.2f}\n"
            f"Recommendation: {'Recommended' if suitability_score > 0.7 else 'Not Recommended'}"
        )

        return summary, urban_data

    except Exception as e:
        return f"Error analyzing {city_name}: {str(e)}", None

# Tool: Elevation Analysis
@tool(response_format="content_and_artifact")
def analyze_elevation(min_elevation: int, max_elevation: int) -> Tuple[str, np.ndarray]:
    """
    Analyze elevation data using a DEM and return suitable areas.
    """
    try:
        dem_path = 'D:/LLM/Lexgis/Lex_Geographica/data/elevation.tif'
        dem_data = rasterio.open(dem_path)
        elevation = dem_data.read(1)

        # Identify suitable areas
        suitable_areas = np.where(
            (elevation >= min_elevation) & (elevation <= max_elevation), 1, 0
        )
        return f"Analysis complete for elevation range {min_elevation}-{max_elevation}m", suitable_areas

    except Exception as e:
        return f"Error analyzing elevation: {str(e)}", None

# Tool: Route Finder
@tool(response_format="content_and_artifact")
def find_route(start_location: str, end_location: str, transport_mode: str = 'walk') -> Tuple[str, Dict[str, Any]]:
    """
    Find the shortest route between two locations.
    """
    try:
        geolocator = Nominatim(user_agent="my_route_finder")
        start_location_data = geolocator.geocode(start_location)
        end_location_data = geolocator.geocode(end_location)

        if not (start_location_data and end_location_data):
            return "Could not find one or both locations", None

        start_coords = (start_location_data.latitude, start_location_data.longitude)
        end_coords = (end_location_data.latitude, end_location_data.longitude)
        north = max(start_coords[0], end_coords[0]) + 0.01
        south = min(start_coords[0], end_coords[0]) - 0.01
        east = max(start_coords[1], end_coords[1]) + 0.01
        west = min(start_coords[1], end_coords[1]) - 0.01

        graph = ox.graph_from_bbox(north, south, east, west, network_type=transport_mode)
        start_node = ox.nearest_nodes(graph, start_coords[1], start_coords[0])
        end_node = ox.nearest_nodes(graph, end_coords[1], end_coords[0])

        route = nx.shortest_path(graph, start_node, end_node, weight='length')

        # Extract route coordinates
        route_coords = [[graph.nodes[node]['y'], graph.nodes[node]['x']] for node in route]
        total_distance = sum(
            geodesic(route_coords[i], route_coords[i+1]).kilometers for i in range(len(route_coords)-1)
        )
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

    except Exception as e:
        return f"Error finding route: {str(e)}", None

# Artifact Extractors
def get_artifact(llm_response, i, tool_function):
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        tool_call = llm_response.tool_calls[i]
        tool_response = tool_function.invoke(tool_call)
        return {
            'content': tool_response.content,
            'artifact': tool_response.artifact,
            'tool_call': tool_call
        }
    return None

def display_route_map(route_data: Dict[str, Any]) -> None:
    center_lat = (route_data['start_coords'][0] + route_data['end_coords'][0]) / 2
    center_lon = (route_data['start_coords'][1] + route_data['end_coords'][1]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add route and markers
    folium.PolyLine(route_data['route_coords'], weight=4, color='blue').add_to(m)
    folium.Marker(route_data['start_coords'], popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(route_data['end_coords'], popup="End", icon=folium.Icon(color='red')).add_to(m)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        m.save(tmp.name)
        st.components.v1.html(open(tmp.name, 'r').read(), height=600)
    Path(tmp.name).unlink()

# Manager Agent Logic
def process_user_request(query: str):
    llm_response = llm_with_tools.invoke(query)
    results = []

    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        for i, tool_call in enumerate(llm_response.tool_calls):
            tool_name = tool_call['name']
            if tool_name == 'analyze_elevation':
                results.append(get_artifact(llm_response, i, analyze_elevation))
            elif tool_name == 'find_route':
                result = get_artifact(llm_response, i, find_route)
                if result and result.get('artifact'):
                    display_route_map(result['artifact'])
                results.append(result)
            elif tool_name == 'urban_suitability_analysis':
                results.append(get_artifact(llm_response, i, urban_suitability_analysis))
    return results

# Streamlit App
st.set_page_config(page_title="Geo Analysis Tools", layout="wide")
st.title("Geographic Analysis Assistant")

@st.cache_resource
def init_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    return llm.bind_tools([analyze_elevation, find_route, urban_suitability_analysis])

llm_with_tools = init_llm()

# query = st.text_input("Enter your query:")
query = st.text_input(
    "Enter your query (e.g., 'Find a route from Central Park, New york to Times Square, New york' or 'Analyze elevation between 500 and 600 meters' or 'Tell me if San Francisco, California is a suitable place for housing development')",
    key="query"
)
if st.button("Process"):
    with st.spinner("Processing..."):
        results = process_user_request(query)
        if results:
            for result in results:
                st.success(result['content'])
                if isinstance(result['artifact'], np.ndarray):
                    min_elev = result['tool_call']['args']['min_elevation']
                    max_elev = result['tool_call']['args']['max_elevation']
                    fig, ax = plt.subplots()
                    ax.imshow(result['artifact'], cmap='Greens')
                    ax.set_title(f"Suitable Areas ({min_elev}-{max_elev}m)")
                    st.pyplot(fig)
        else:
            st.error("No results generated.")
