import osmnx as ox
from typing import Tuple, Dict
from populationthing import calculate_suitability
from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def urban_suitability_analysis(city_name: str) -> Tuple[str, Dict[str, any]]:
    """
    Analyze urban suitability for housing development.
    
    Args:
        city_name (str): Name of the city or locality to analyze.
    
    Returns:
        Tuple[str, Dict[str, any]]: Analysis summary and detailed urban suitability data.
    """
    try:
        ox.config(use_cache=True, log_console=True)
        
        gdf_city = ox.geocode_to_gdf(city_name)
        graph = ox.graph_from_place(city_name, network_type='all')
        
        buildings = ox.geometries_from_place(city_name, tags={'building': True})
        building_areas = buildings.area
        total_building_area = building_areas.sum()
        avg_building_area = building_areas.mean()
        
        city_area = gdf_city.iloc[0].geometry.area
        population_density = total_building_area / city_area
        
        street_lengths = ox.graph_to_gdfs(graph, nodes=False, edges=True)['length']
        total_street_length = street_lengths.sum()
        
        suitability_score = calculate_suitability(
            population_density,
            avg_building_area,
            total_street_length,
            city_area
        )
        
        urban_data = {
            'city': city_name,
            'total_area': city_area,
            'population_density': population_density,
            'avg_building_area': avg_building_area,
            'total_building_area': total_building_area,
            'total_street_length': total_street_length,
            'suitability_score': suitability_score,
            'recommendation': 'Recommended' if suitability_score > 0.56 else 'Not Recommended'
        }
        
        summary = (
            f"Urban Suitability Analysis for {city_name}\n"
            f"Suitability Score: {suitability_score:.2f}\n"
            f"Recommendation: {'Recommended' if suitability_score > 0.56 else 'Not Recommended'}"
        )
        
        return summary, urban_data
    
    except Exception as e:
        return f"Error analyzing {city_name}: {str(e)}", None
