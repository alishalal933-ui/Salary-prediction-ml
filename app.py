import streamlit as st
import joblib
import numpy as np

# Model load kar
model = joblib.load('salary_model.pkl')

st.title("Salary Prediction App 💰")
st.write("Experience aur Age daal ke salary predict karo")

# User input
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0)
age = st.number_input("Age", min_value=18, max_value=65, value=25)

# Predict button
if st.button("Predict Salary"):
    features = np.array([[experience, age]])
    prediction = model.predict(features)
    st.success(f"Estimated Salary: ₹{prediction[0]:,.2f}")
