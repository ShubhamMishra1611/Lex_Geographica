# tools/elevation_tool.py
import rasterio
import numpy as np
from typing import Tuple
from langchain_core.tools import tool

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
    dem_path = 'data/elevation.tif'  # Update path in settings
    dem_data = rasterio.open(dem_path)
    elevation = dem_data.read(1)
    
    suitable_areas = np.where(
        (elevation >= min_elevation) & (elevation <= max_elevation), 
        1, 
        0
    )
    
    return f"Analysis complete for elevation range {min_elevation}-{max_elevation}m", suitable_areas

# tools/route_tool.py
from typing import Dict, Any, Tuple
from langchain_core.tools import tool
from ..utils.geo_utils import get_coordinates, calculate_route
from ..utils.map_utils import create_route_map

@tool(response_format="content_and_artifact")
def find_route(
    start_location: str,
    end_location: str,
    transport_mode: str = 'walk'
) -> Tuple[str, Dict[str, Any]]:
    """
    Find and visualize the shortest route between two locations.
    """
    # Get coordinates
    start_coords = get_coordinates(start_location)
    end_coords = get_coordinates(end_location)
    
    if not (start_coords and end_coords):
        return "Could not find one or both locations", None
    
    route_data = calculate_route(
        start_coords, 
        end_coords, 
        transport_mode,
        start_location,
        end_location
    )
    
    if not route_data:
        return "No route found between the specified locations", None
    
    return f"Route found! Distance: {route_data['total_distance']:.2f} km", route_data