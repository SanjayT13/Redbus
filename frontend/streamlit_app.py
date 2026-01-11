import streamlit as st 
import requests
import pandas as pd
import json

FASTAPI_URL = "http://127.0.0.1:8000/api/predict"

st.set_page_config(page_title= "Redbus demand prediction", layout="centered")
st.title("Redbus number of travellers Prediction")
st.write("Enter the route details below to get the predicted number of travellers.")

st.subheader("Input Data")

df = st.data_editor(pd.DataFrame({"route_key": ["2025-02-11_46_45"],
            "doj": ["2025-02-11"],
            "srcid": [46],
            "destid": [45]}),
            num_rows="dynamic",
            width = "stretch")

if st.button("Get Predictions"): 
    payload = {"route_key": df["route_key"].tolist(),
            "doj": df["doj"].tolist(),
            "srcid": df["srcid"].tolist(),
            "destid": df["destid"].tolist()}

if "payload" in locals(): 
    with st.spinner("Fetching predictions..."):
        try:
            response = requests.post(FASTAPI_URL, json=payload)
            response.raise_for_status()

            if response.status_code == 200:
                result = response.json()
                st.success("Predictions fetched successfully!")
                st.subheader("Predicted Number of Travellers")
                st.json(result)
            else:
                st.error(f"Failed to get predictions. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
