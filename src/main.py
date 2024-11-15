import streamlit as st
from tools.elevation_tool import analyze_elevation
from tools.roads_tool import analyze_roads

# Streamlit UI
st.title("Urban Planning Map Data Analysis")

# File inputs
dem_path = st.file_uploader("Upload DEM File", type=["tif"])
road_path = None #st.file_uploader("Upload Road Shapefile", type=["shp"])

if dem_path:
    st.write("Analyzing elevation data...")
    elevation_result = analyze_elevation(dem_path)

    # Display output and download link for plot
    st.text(elevation_result["message"])
    st.image(elevation_result["plot_path"])
    
if road_path:
    st.write("Analyzing road infrastructure...")
    road_result = analyze_roads(road_path)
    st.text(road_result["message"])


# if __name__ == '__main__':
#     pass