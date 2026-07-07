import streamlit as st
import joblib
import numpy as np

model = joblib.load('salary_model.pkl')

st.title("💰 Salary Prediction App")
st.write("Exp aur Age daal kar salary predict karo")

exp = st.number_input("Experience in Years", min_value=0, max_value=50, value=5)
age = st.number_input("Age", min_value=18, max_value=70, value=25)

if st.button("Predict Salary"):
    input_data = np.array([[exp, age]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary: Rs {prediction[0]:,.2f}")
