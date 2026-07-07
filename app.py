import streamlit as st
import joblib
import numpy as np

model = joblib.load('salary_model.pkl')

st.title("Salary App")
exp = st.number_input("Exp")
if st.button("Predict"):
    input_data = np.array([[exp]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Salary: {prediction[0]}")
