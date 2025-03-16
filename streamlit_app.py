import streamlit as st
import numpy as np
import pickle

# Load the trained model
model_path = "diabetes_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Model file 'diabetes_model.pkl' not found! Please upload it to the same directory as this script.")
    st.stop()

# Streamlit UI
st.title("🔬 Diabetes Prediction App")
st.write("Enter the required details below and click **Predict** to see the results.")

# User input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=500, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=10, max_value=100, value=30)

# Convert user inputs to numpy array
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(user_input)[0]
    
    if prediction == 1:
        st.error("⚠️ High risk of Diabetes! Consult a doctor.")
    else:
        st.success("✅ No signs of Diabetes detected! Stay healthy. 🍏")

# Footer
st.write("💡 This is a simple ML-based diabetes prediction system. Always consult a doctor for medical advice.")
