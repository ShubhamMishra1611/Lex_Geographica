import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
def analyze_elevation(min_elevation: int=100, 
                      max_elevation: int=200, 
                      output_img_name: str='output_elevation.png')-> Dict[str, Any]:
    """
    Analyze elevation data using DEM and return summary statistics and file path of plot.

    Args:
        min_elevation int: Minimum elevation threshold.
        max_elevation int: Maximum elevation threshold.
        output_img_name str: Name of the plot image.
    Returns:
        dict: Dictionary containing summary statistics and plot path.
    """
    # Load DEM data
    output_img_name = 'output_elevation.png'
    dem_path = '/home/stemmets/Desktop/projllm/lexgis/Lex_Geographica/data/elevation.tif'
    print(f"LOGG: {min_elevation = }; {max_elevation = }")
    dem_data = rasterio.open(dem_path)
    elevation = dem_data.read(1)

    # Mask for suitable elevation range
    suitable_areas = np.where((elevation >= min_elevation) & (elevation <= max_elevation), 1, 0)

    # Calculate total suitable area (assuming each pixel represents area in square meters)
    pixel_area = dem_data.res[0] * dem_data.res[1]  # Resolution of each pixel
    total_suitable_area = np.sum(suitable_areas) * pixel_area  # in square meters

    # Convert to square kilometers
    total_suitable_area_km2 = total_suitable_area / 1e6

    # Save a plot of suitable areas (optional visualization)
    plt.figure(figsize=(10, 10))
    plt.imshow(suitable_areas, cmap='Greens', extent=dem_data.bounds)
    plt.title(f'Suitable Areas for Development ({min_elevation}m-{max_elevation}m elevation)')
    plt.colorbar(label='1 = Suitable, 0 = Not Suitable')
    plt.savefig(output_img_name)
    plt.close()

    # Return structured data for LLM
    return {
        "total_suitable_area_km2": total_suitable_area_km2,
        "plot_path": output_img_name,
        "message": f"Total area suitable for development is approximately {total_suitable_area_km2:.2f} km²."
    }
