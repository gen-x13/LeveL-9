# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 09:23:32 2026

@author: GenXCode
"""

# FFT Unsupervised Learning Web App

# Pandas for dataset handling
import pandas as pd

# Plotly for visualization
import plotly.express as px

# Streamlit for the web app
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

# BirdSong Model
from birds_sound_model import BirdModel

# Paths
from pathlib import Path

# Path to the dataset example "bird sound"
data_path = Path(__file__).parent / "data" / "bird_sound_data.csv"

# Base Path for audios
base_path = Path(__file__).parent

# -------------------------------   Menu Params   --------------------------- #        
        
# Page Icon, side bar collpase
st.set_page_config(page_title="Clustering Sound And More...", 
                   initial_sidebar_state="collapsed")

# Horizontal menu
selected=option_menu(
        menu_title="Menu",
        options = ["BirdSong Example", "Music"], 
        icons = ["music-note-list", "soundwave"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
)   

# -------------------------------   Bird's data   --------------------------- #

# Security preventing any reading problem and any cache data problem
@st.cache_data(show_spinner="Loading bird's sound data...")
def load_data():
    return pd.read_csv(data_path)

data = load_data()

def extract_relative_path(full_path):
    path = Path(full_path)
    # Find "sample.." in the part of the strings in the path
    parts = path.parts
    try:
        idx = parts.index("sample_audios_xeno_canto")
        # Rebuild the path with the index
        return str(Path(*parts[idx:]))
    except ValueError:
        # If not found, return the file's name
        return f"sample_audios_xeno_canto/{path.name}"

# Apply the transformation inside the dataset
data['Audio'] = data['Audio'].apply(extract_relative_path)

# Save the modifications in the csv data file
data.to_csv(data_path, index=False)

print(data['Audio'][0])

# ---------------------------------- Model ---------------------------------- #

# For every object : st.cache_resource -> avoid problems with cache later

# BirdSong Clustering Model
#@st.cache_data # only cache for objects
def cluster_birdsong(data, clusters):
    birdsong = BirdModel(data, clusters)
    return birdsong.run()


# --------------------------------- BirdSong -------------------------------- #


# BirdSong Model Example Page Selection :
if selected == "BirdSong Example": 
    
    st.title("BirdSong Clustering Example")
    
    # Explication de clustering
    # Explication de l'utilisation de l'app
    # Possibilité de choisir entre 3 à 9 clusters 
    # Audio intégré après chaque prédiction (par groupe)
    
    st.header("What is clustering ?")
    # HTML Writing to reduce the space between two lines
    st.markdown("Clustering is one of the few ML methods of prediciton.<br>It consists of grouping data points based on the nearest point.<br>This is an example of clustering with BirdSongs.<br>You can choose how many clusters you want with the ~400 birdsongs.",
                unsafe_allow_html=True)
    
    num_cluster = st.selectbox( "Number of clusters ?", options= range(3, 10),
                            index=0)
    
    if num_cluster is not None:
        with st.spinner("Prediction in progress..."):
            df, pca_data = cluster_birdsong(data, num_cluster)

            # Creating a dataframe from the components for plotly 3D
            # visualization
            df_plot = pd.DataFrame({
                                    # Components (array)
                                    'PC1':pca_data[:,0],
                                    'PC2':pca_data[:,1],
                                    'PC3':pca_data[:,2],
                                    # Informations
                                    'Names':df['Name'],
                                    'Species':df['Species'],
                                    # Paths Audio Files 
                                    'Audios':df["Audio"],
                                    # Cluster Labels
                                    'Clusters':df['Clusters'],
                                    
                                    
                                    })
        
        # 3D Scatter Plot of the results
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                            color="Clusters", 
                            color_continuous_scale='Plasma',
                            hover_data=['Names', 'Species'],
                            title="BirdSongs 3D Clustering")
        
        # Adjusting the size and opacity of the points
        fig.update_traces(
                marker=dict(
                        size=1.5, # 0.5 gives a space vibe, might use it elsewhere
                        opacity=0.8
                    )
            )
        
        # Improving the view of the 3D Graph
        fig.update_layout(
                scene=dict(
                        xaxis_title='PC1',
                        yaxis_title='PC2',
                        zaxis_title='PC3'
                    ),
                width=500,
                height=500
            )
        
        st.plotly_chart(fig)
            
        # Subheader
        st.header("BirdSongs Groups")
        
            # Each Group -> Labeled Group cluster_uni.count()[0] etc
            # Each Group -> Audios Samples 
            # Each Name & Species (& maybe a picture)
        
        for cluster in df['Clusters'].unique():
            
            st.subheader(f"Cluster {cluster}")
            
            # Filtering audios from this cluster
            cluster_data = df[df['Clusters']==cluster]
            
            # Using iterrows for EACH rows
            for index, row in cluster_data.iterrows():
                
                st.markdown(f"**{row['Name']}** - *{row['Species']}*")
                audio_path = row["Audio"]
                st.write(audio_path)
                st.write(audio_path.exists())
                
                #st.audio(audio_path, format="audio/mpeg", loop=True)

    else:
        st.warning("Please, make a selection.")
   
# --------------------------------  Music Page  ----------------------------- #

elif selected == "Music":
    
    st.title("Music Mashup Clusters")
    
    st.caption("🏗 It's still under construction, come back in a few days")
