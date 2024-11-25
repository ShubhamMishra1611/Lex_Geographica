import osmnx as ox
import geopandas as gpd
import numpy as np
from langchain_core.tools import tool



def analyze_urban_suitability(city_name):
    """
    Analyze urban suitability for housing development
    
    Parameters:
    city_name (str): Name of the city or locality to analyze
    
    Returns:
    dict: Comprehensive urban analysis report
    """
    # Set up the analysis
    ox.config(use_cache=True, log_console=True)
    
    # Retrieve city data
    try:
        # Fetch city boundaries and street network
        gdf_city = ox.geocode_to_gdf(city_name)
        graph = ox.graph_from_place(city_name, network_type='all')
        
        # Calculate population density
        buildings = ox.geometries_from_place(city_name, tags={'building': True})
        
        # Analyze building characteristics
        building_areas = buildings.area
        total_building_area = building_areas.sum()
        avg_building_area = building_areas.mean()
        
        # Calculate population density estimate
        city_area = gdf_city.iloc[0].geometry.area
        population_density = total_building_area / city_area
        
        # Analyze street network
        street_lengths = ox.graph_to_gdfs(graph, nodes=False, edges=True)['length']
        total_street_length = street_lengths.sum()
        
        # Assess development suitability factors
        suitability_score = calculate_suitability(
            population_density, 
            avg_building_area, 
            total_street_length, 
            city_area
        )
        
        return {
            'city': city_name,
            'total_area': city_area,
            'population_density': population_density,
            'avg_building_area': avg_building_area,
            'total_building_area': total_building_area,
            'total_street_length': total_street_length,
            'suitability_score': suitability_score,
            'recommendation': 'Recommended' if suitability_score > 0.7 else 'Not Recommended'
        }
    
    except Exception as e:
        print(f"Error analyzing {city_name}: {e}")
        return None

def calculate_suitability(density, avg_building_size, street_length, city_area):
    """
    Calculate a comprehensive suitability score for housing development
    
    Parameters:
    density (float): Population density
    avg_building_size (float): Average building area
    street_length (float): Total street length
    city_area (float): Total city area
    
    Returns:
    float: Suitability score between 0 and 1
    """
    # Normalize and weight different factors
    density_factor = min(density / 1000, 1)  # Normalize density
    building_size_factor = 1 - min(avg_building_size / 500, 1)  # Prefer moderate building sizes
    infrastructure_factor = min(street_length / (city_area * 100), 1)
    
    # Weighted calculation (adjust weights as needed)
    suitability_score = (
        0.4 * density_factor + 
        0.3 * building_size_factor + 
        0.3 * infrastructure_factor
    )
    
    return suitability_score

# Example usage
def main():
    city = "San Francisco, California"
    analysis_result = analyze_urban_suitability(city)
    
    if analysis_result:
        print("\nUrban Development Analysis Report:")
        for key, value in analysis_result.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()

# Required dependencies:
# pip install osmnx geopandas matplotlib