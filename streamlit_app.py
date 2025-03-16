import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    model = pickle.load(open("classifier.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model file not found. Please check the file path.")
    st.stop()

# Title of the app
st.title("Diabetes Prediction App")

# Collect user input for 8 features
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Predict button
if st.button("Predict"):
    # Combine user input into an array
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    # Manually apply Standard Scaling (assuming mean & std from training)
    mean_values = np.array([3.8, 120, 70, 20, 80, 32, 0.47, 33])  # Example means
    std_values = np.array([3.4, 30, 12, 8, 100, 5, 0.33, 12])  # Example standard deviations
    input_scaled = (features - mean_values) / std_values

    # Perform prediction
    prediction = model.predict(input_scaled)

    # Show result
    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes! Consult a doctor.")
    else:
        st.success("✅ No Diabetes Risk! Stay healthy.")
