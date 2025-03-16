import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open("classifier.pkl", "rb"))

# Sample scaler (Replace with actual scaler used during training)
scaler = StandardScaler()

# User inputs
glucose = st.number_input("Glucose Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)
# (Add other inputs)

features = [glucose, bmi, age]  # Add all required features

# Debug: Print raw inputs
st.write(f"User Input Features: {features}")

# Check if input scaling is needed
input_scaled = scaler.fit_transform([features])  # Replace with the original scaler used
st.write(f"Scaled Features: {input_scaled}")

# Prediction
prediction = model.predict(input_scaled)  # Use scaled inputs
st.write(f"Raw Model Prediction: {prediction}")

# Display result
if prediction[0] == 1:
    st.error("High Risk of Diabetes")
else:
    st.success("No Diabetes Risk")
