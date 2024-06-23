
import requests
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load the model and scaler
model = joblib.load('dt.pkl')
scaler = joblib.load('Fifapredict.pkl')

# Feature columns
cols = ['overall', 'movement_reactions', 'potential', 'passing', 'wage_eur',
        'mentality_composure', 'value_eur', 'dribbling']

# Function to load lottie animations (if needed)
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.header("Player Rating Prediction")

with st.container():
    # Input sliders
    overall = st.slider('Overall', 0, 100, 0)
    potential = st.slider('Potential', 0, 100, 0)
    dribbling = st.slider('Dribbling', 0, 100, 0)
    composure = st.slider('Composure', 0, 100, 0)
    wage_eur = st.slider('Wage (EUR)', 0, 1_000_000, 0)  # Adjust max value as needed
    passing = st.slider('Passing', 0, 100, 0)
    movement_reactions = st.slider('Movement Reactions', 0, 100, 0)
    value_eur = st.slider('Player Value (EUR)', 0, 100_000_000, 0)  # Adjust max value as needed

    # Collect responses
    responses = [overall, movement_reactions, potential, passing, wage_eur,
                 composure, value_eur, dribbling]

    submit = st.button("Submit")

    if submit:
        # Create DataFrame from user inputs
        responseDF = pd.DataFrame([responses], columns=cols)
        
        # Display the raw input data
        st.write("Raw input data:")
        st.write(responseDF)
        
        # Scale the data
        scaledData = scaler.transform(responseDF)
        
        # Convert scaled data back to DataFrame for easier manipulation
        scaledDF = pd.DataFrame(scaledData, columns=cols)
        
        # Drop the 'overall' column
        scaledDF = scaledDF.drop(columns=['overall'])
        
        # Display the scaled data before prediction
        st.write("Scaled data (excluding 'overall'):")
        st.write(scaledDF)
        
        # Predict using the model
        predict = model.predict(scaledDF)
        
        # Display the prediction
        st.subheader("Player rating = " + str(round(predict[0])))
