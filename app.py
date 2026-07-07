import streamlit as st
import joblib
import numpy as np

model = joblib.load('salary_model.pkl')

st.title("Salary App")

# Model ko kitne features chahiye ye pata karo
n_features = model.n_features_in_
st.write(f"Model expects {n_features} inputs")

exp = st.number_input("Exp")
age = st.number_input("Age")

if st.button("Predict"):
    if n_features == 1:
        input_data = np.array([[exp]])
    else:
        input_data = np.array([[exp, age]]) # 2 features bhej do
    
    prediction = model.predict(input_data)
    st.write(f"Predicted Salary: {prediction[0]:.2f}")
