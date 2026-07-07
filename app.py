import streamlit as st
import joblib
import numpy as np

model = joblib.load('salary_model.pkl')

st.title("Salary App")
exp = st.number_input("Exp")
age = st.number_input("Age") # 2nd feature add kar diya

if st.button("Predict"):
    input_data = np.array([[exp, age]]) # 2 features bhej diye
    prediction = model.predict(input_data)
    st.write(f"Predicted Salary: {prediction[0]:.2f}")
