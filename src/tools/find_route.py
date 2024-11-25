import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from typing import Tuple, Dict
from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def find_route(
    start_location: str,
    end_location: str,
    transport_mode: str = 'walk'
) -> Tuple[str, Dict[str, any]]:
    """
    Find the shortest route between two locations.
    
    Args:
        start_location (str): Starting location name or address
        end_location (str): Destination location name or address
        transport_mode (str): Mode of transport ('walk', 'drive', 'bike')
        
    Returns:
        Tuple[str, Dict[str, Any]]: Content message and route data
    """
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
    
    graph = ox.graph_from_bbox(north, south, east, west, network_type=transport_mode)
    start_node = ox.nearest_nodes(graph, start_coords[1], start_coords[0])
    end_node = ox.nearest_nodes(graph, end_coords[1], end_coords[0])
    
    try:
        route = nx.shortest_path(graph, start_node, end_node, weight='length')
    except nx.NetworkXNoPath:
        return "No route found between the specified locations", None
    
    route_coords = [
        [graph.nodes[node]['y'], graph.nodes[node]['x']] for node in route
    ]
    
    total_distance = sum(
        geodesic(route_coords[i], route_coords[i + 1]).kilometers
        for i in range(len(route_coords) - 1)
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
