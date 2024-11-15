import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def analyze_elevation(dem_path, min_elevation=100, max_elevation=200, output_plot_path='output_elevation.png'):
    """
    Analyze elevation data and return summary statistics and file path of plot.

    Args:
        dem_path (str): Path to the DEM file.
        min_elevation (int): Minimum elevation threshold.
        max_elevation (int): Maximum elevation threshold.
        output_plot_path (str): File path to save the plot image.

    Returns:
        dict: Dictionary containing summary statistics and plot path.
    """
    # Load DEM data
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
    plt.savefig(output_plot_path)
    plt.close()

    # Return structured data for LLM
    return {
        "total_suitable_area_km2": total_suitable_area_km2,
        "plot_path": output_plot_path,
        "message": f"Total area suitable for development is approximately {total_suitable_area_km2:.2f} km²."
    }
