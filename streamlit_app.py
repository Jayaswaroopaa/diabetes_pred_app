import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
try:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Please check the file path.")
    st.stop()

# Title of the app
st.title("Diabetes Prediction App")

# Collect user input
glucose = st.number_input("Enter Glucose Level", min_value=0, max_value=300, step=1)
bmi = st.number_input("Enter BMI", min_value=0.0, max_value=100.0, step=0.1)
age = st.number_input("Enter Age", min_value=0, max_value=120, step=1)

# Combine user input into a NumPy array
features = np.array([[glucose, bmi, age]])

# Manually apply Standard Scaling (assuming mean & std from training)
# Set these values based on the dataset used for training
mean_values = np.array([120, 32, 33])  # Example means for glucose, BMI, and age
std_values = np.array([30, 5, 12])  # Example standard deviations

input_scaled = (features - mean_values) / std_values  # Apply standardization
st.write("🔍 Scaled Features:", input_scaled)

# Perform prediction
try:
    prediction = model.predict(input_scaled)
    st.write("🔍 Model Output:", prediction)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes! Consult a doctor.")
    else:
        st.success("✅ No Diabetes Risk! Stay healthy.")
except Exception as e:
    st.error(f"⚠ Prediction Error: {str(e)}")
