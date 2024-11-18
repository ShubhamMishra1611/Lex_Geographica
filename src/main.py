import streamlit as st
from tools.elevation_tool import analyze_elevation
from tools.roads_tool import analyze_roads
from agent.urban_agent import DEM_agent
import os

import streamlit as st

st.title("Urban Planning Map Data Analysis")

# File inputs
dem_path = st.file_uploader("Upload DEM File", type=["tif"])
# road_path = st.file_uploader("Upload Road Shapefile", type=["shp"])

# Query input
query = st.text_input("Enter your query (e.g., 'Analyze elevation', 'Check road connectivity')")

if query:
    st.write(f"Processing query: {query}")

    if "elevation" in query.lower() and dem_path:
        st.write("Analyzing elevation data...")
        agent_guy = DEM_agent(user_query=query)

        if os.path.exist("output_elevation.png"):
            st.image("output_elevation.png")

        # elevation_result = analyze_elevation(dem_path)

        # Display output and download link for plot
        # st.text(elevation_result["message"])
        # st.image(elevation_result["plot_path"])

    # elif "road" in query.lower() and road_path:
    #     st.write("Analyzing road infrastructure...")
    #     road_result = analyze_roads(road_path)
    #     st.text(road_result["message"])
    else:
        st.warning("Please upload the relevant file(s) for your query.")
else:
    st.info("Enter a query to begin analysis.")
