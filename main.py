# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 09:23:32 2026

@author: GenXCode
"""

# FFT Unsupervised Learning Web App

# Randomizing the results
import random

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

import requests
from urllib.parse import quote

# Path to the dataset example "bird sound"
data_path = Path(__file__).parent / "data" / "bird_sound_data.csv"

# Base Path for audios
base_path = Path(__file__).parent

# -------------------------------   Menu Params   --------------------------- #        
        
# Page Icon, side bar collpase
st.set_page_config(page_title="Clustering Sound And More...", 
                   initial_sidebar_state="collapsed",
                  layout="wide")
                   #layout="wide")

# Horizontal menu
selected=option_menu(
        menu_title="Menu",
        options = ["Wildlife", "Music"], 
        icons = ["music-note-list", "soundwave"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
)   

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.tenor.com/KzdfR6Hek7YAAAAM/forest-rain.gif");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------   Initialization   --------------------------- #
if "show_cluster" not in st.session_state:
    st.session_state.show_cluster = True
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashbaord = False

# -------------------------------   Bird's data   --------------------------- #

# Security preventing any reading problem and any cache data problem
#@st.cache_data(show_spinner="Loading bird's sound data...")
def load_data():
    return pd.read_csv(data_path)
data = load_data()                                              

# ---------------------------------- Model ---------------------------------- #

# For every object : st.cache_resource -> avoid problems with cache later

# BirdSong Clustering Model
def cluster_birdsong(data, clusters):
    birdsong = BirdModel(data, clusters)
    return birdsong.run()
        
@st.cache_data
def display_prediction(data, num_cluster):
        df, pca_data = cluster_birdsong(data, num_cluster)
        return df, pca_data

def data_pca(df, pca):

        data = pd.DataFrame({
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
        return data


@st.cache_data
def get_bird_image(species_name):

    url = "https://api.inaturalist.org/v1/taxa"

    params = {
        "q": species_name,
        "limit": 1
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        results = data.get("results")

        if results:

            photo = results[0].get("default_photo")

            if photo:
                return photo.get("medium_url")

    except Exception:
        pass

    return None

       


# --------------------------------- BirdSong -------------------------------- #


# BirdSong Model Page Selection :  

if selected == "Wildlife": 
    
    st.title("Xeno Canto Wildlife Song Clustering")
    st.caption("⚠️ WIP ⚠️")
        
    st.markdown("<b><p style='color:black;font-size:20px;background:rgba(255,255,255,0.20);border-radius:12px;padding:10px 14px;display:inline;backdrop-filter:blur(6px);'>This app groups birds based on their birdsong. Select how many clusters you want to observe within the ~400 audios from Xeno Canto.</p></b>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    # Buttons from clustering to dashboard
    with col2:
        if st.button("Clustering") :
                st.session_state.show_cluster = True
                st.session_state.show_dashboard = False
        if st.button("Dashboard") :
                st.session_state.show_dashboard = True
                st.session_state.show_cluster = False
              
    with col1:
        if st.session_state.show_cluster :
                num_cluster = st.slider("Number of clusters?", 3, 10, 3)
                if num_cluster is not None:
                    with st.spinner("Prediction in progress..."):
                            
                        df, pca_data = display_prediction(data, num_cluster)
                        # Creating a dataframe from the components for plotly 3D
                        # visualization
                        df_plot = data_pca(df, pca_data)
                            
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
                    st.header("BirdSongs Groups Samples")
                
                    for cluster in sorted(df['Clusters'].unique()):
                
                        st.subheader(f"Cluster {cluster}")
                
                        # Filtering audios from this cluster
                        cluster_data = df[df['Clusters']== cluster].sort_values(by="Name")
                
                        # Randomizing the index
                        index = random.randrange(len(cluster_data))
                        # Take only one audio from the random index
                        row = cluster_data.iloc[index]
                        # Audio Path
                        audio_path = base_path / row["Audio"]
                        # Display name and audio
                        st.markdown(f"**{row['Name']}** - *{row['Species']}*")
                        st.audio(audio_path, format="audio/mpeg", loop=False)
                
                else:
                    st.warning("Please, make a selection.") 

        elif st.session_state.show_dashboard and not st.session_state.show_cluster :
                # Storing Data
                df, pca_data = display_prediction(data, 5)
                # Sorting Values
                df = df.sort_values(by="Name", ascending=True)
                # Creating a new column combining name and species
                df["animal-specie"] = df["Name"] + " - " + df["Species"]

                # Selection of birds
                spe_bird_sel = st.selectbox(
                    "Select a animal",
                    df['animal-specie'],
                    index=None,
                    placeholder="Select a bird.",
                    accept_new_options=True,
                )
                #with st.spinner("Prediction in progress..."):
                if spe_bird_sel is not None:

                        subcol1, subcol2 = st.columns(2)
                        selected_row = df[df["animal-specie"] == spe_bird_sel].iloc[0]
        
                        with subcol1:
                                st.caption("🏗 It's still under construction, come back in a few days")
                                # 5 samples with strong similarities (points proches)
                                

                        with subcol2:
                                st.caption("🏗 It's still under construction, come back in a few days")
                                # Same specie different cluster or picture of the bird or galerie for the specie
                                import requests

                                bird_name = selected_row["Species"]
                                
                                # Display
                                image_url = get_bird_image(bird_name)
                                
                                if image_url:
                                        st.image(image_url, width=300)
                                        # Take only one audio from the random index
                                        row = df[df["animal-specie"] == spe_bird_sel].iloc[0]
                                        # Audio Path
                                        audio_path = base_path / row["Audio"]
                                        # Display name and audio
                                        st.markdown(f"**{row['Name']}** - *{row['Species']}*")
                                        st.audio(audio_path, format="audio/mpeg", loop=True)
                                
                                else:
                                        st.write("Image non trouvée")


                else:
                        st.warning("Select an animal")




# --------------------------------  Music Page  ----------------------------- #

elif selected == "Music":
    
    st.title("Music Mashup Clusters")
    
    st.caption("🏗 It's still under construction, come back in a few days")



























