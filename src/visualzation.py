# visualization.py
import folium
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from pathlib import Path
import tempfile
import streamlit as st


def plot_suitable_areas(suitable_areas: np.ndarray, min_elevation: int, max_elevation: int):
    """
    Create and return a matplotlib figure of suitable areas based on elevation.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(suitable_areas, cmap='Greens')
    plt.colorbar(im, ax=ax, label='1 = Suitable, 0 = Not Suitable')
    ax.set_title(f'Suitable Areas ({min_elevation}m-{max_elevation}m elevation)')
    return fig


def display_route_map(route_data: Dict[str, Any]) -> None:
    """
    Create and display a Folium map with the route.
    """
    if not route_data:
        st.error("No route data available")
        return

    # Center map on the midpoint of start and end coordinates
    center_lat = (route_data['start_coords'][0] + route_data['end_coords'][0]) / 2
    center_lon = (route_data['start_coords'][1] + route_data['end_coords'][1]) / 2

    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add start and end markers
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

    # Draw the route
    folium.PolyLine(
        route_data['route_coords'],
        weight=4,
        color='blue',
        opacity=0.8
    ).add_to(m)

    # Save to a temporary file for display
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        m.save(tmp.name)
        st.components.v1.html(open(tmp.name, 'r').read(), height=600)

    # Clean up temporary file
    Path(tmp.name).unlink()
