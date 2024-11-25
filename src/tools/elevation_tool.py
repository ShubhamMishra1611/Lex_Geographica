import rasterio
import numpy as np
from typing import Tuple
from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def analyze_elevation(min_elevation: int, max_elevation: int) -> Tuple[str, np.ndarray]:
    """
    Analyze elevation data using DEM and return suitable areas.
    
    Args:
        min_elevation int: Minimum elevation threshold.
        max_elevation int: Maximum elevation threshold.
        
    Returns:
        Tuple[str, np.ndarray]: Content message and suitable areas array
    """
    dem_path = 'data/elevation.tif'  # Ensure the path to DEM is correct
    dem_data = rasterio.open(dem_path)
    elevation = dem_data.read(1)
    
    suitable_areas = np.where(
        (elevation >= min_elevation) & (elevation <= max_elevation), 
        1, 
        0
    )
    
    return f"Analysis complete for elevation range {min_elevation}-{max_elevation}m", suitable_areas
