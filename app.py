import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading trained model
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = load_model('house_price_model.h5')  # Replace with your actual model filename

# Title and introduction
st.title("🏠 House Price Prediction App")
st.markdown("""
This app predicts **house prices** based on key features using a trained **Neural Network** model.  
Just enter the values below and click **Predict**.
""")

# User input fields
sqft = st.slider("Square Feet (SqFt)", 500, 5000, 1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
brick = st.selectbox("Is the house made of Brick?", ['Yes', 'No'])

# Feature engineering based on inputs
luxury_index = (bedrooms + bathrooms) * (1 if brick == 'Yes' else 0)
price_per_sqft = 150  # Assume a fixed price per sqft or allow user input

# Combine features into array
input_data = np.array([[sqft, bedrooms, bathrooms, luxury_index, price_per_sqft]])

# Normalize using the same scaler parameters (simulating MinMaxScaler)
# You must use the same parameters used during training
scaler = MinMaxScaler()
scaler.min_ = np.array([0, 0, 0, 0, 0])
scaler.scale_ = np.array([1/5000, 1/10, 1/10, 1/40, 1/500])  # Sample scale
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0][0]
    st.success(f"💰 Predicted House Price (normalized): {prediction:.2f}")
    # If model was trained on actual prices (not normalized), multiply to get real value
    approx_price = prediction * 500000  # Assumed max price for rescaling
    st.subheader(f"🏡 Approximate Price: ${approx_price:,.0f}")
