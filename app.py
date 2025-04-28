# app.py

import streamlit as st
import numpy as np
import joblib

# Load the models and scaler
@st.cache_resource
def load_assets():
    lr_model = joblib.load("linear_model.pkl")
    return lr_model
lr_model = load_assets()

# Web UI title
st.title("💰 Bitcoin Closing Price Prediction")
st.subheader("Enter Bitcoin metrics to predict the closing price using ML models")

# User input
open_price = st.number_input("🔓 Open Price", min_value=0.0, value=50000.0, step=100.0)
high_price = st.number_input("📈 High Price", min_value=0.0, value=51000.0, step=100.0)
low_price = st.number_input("📉 Low Price", min_value=0.0, value=49000.0, step=100.0)
volume = st.number_input("📊 Volume", min_value=0.0, value=1_000_000_000.0, step=1_000_000.0)

# Predict button
if st.button("📌 Predict Closing Price"):
    # Prepare and scale input
    user_input = np.array([[open_price, high_price, low_price, volume]])
 

    # Make predictions
    lr_prediction = lr_model.predict(user_input)[0]
   
    # Display predictions
    st.success(f"📈Predicted Close Price: **${lr_prediction:,.2f}**")

