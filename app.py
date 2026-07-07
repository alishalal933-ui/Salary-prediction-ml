import streamlit as st
import pickle
model=pickle.load(open('salary_model.pkl','rb'))
st.title("Salary App")
exp=st.number_input("Exp")
if st.button("Predict"):st.write(model.predict([[exp]])[0])
