import streamlit as st
import joblib

model = joblib.load('salary_model.pkl')

st.title("Salary App")
exp = st.number_input("Exp")
if st.button("Predict"):
    st.write(model.predict([[exp]]))
