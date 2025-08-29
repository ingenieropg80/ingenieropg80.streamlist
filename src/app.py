import streamlit as st
from pickle import load
import pandas as pd

fixed_acidity = st.number_input("fixed acidity", min_value=4.6, max_value=15.9)
volatile_acidity = st.number_input("volatile acidity", min_value=0.12, max_value=1.58)
citric_acid = st.number_input("citric acid", min_value=0.0, max_value=1.0)
residual_sugar = st.number_input("residual sugar", min_value=0.9, max_value=15.5)
chlorides = st.number_input("chlorides", min_value=0.012, max_value=0.611)
free_sulfur_dioxide = st.number_input("free sulfur dioxide", min_value=1.0, max_value=72.0)
total_sulfur_dioxide = st.number_input("total sulfur dioxide", min_value=6.0, max_value=289.0)
density = st.number_input("density", min_value=0.99007, max_value=1.00369)
pH = st.number_input("pH", min_value=2.74, max_value=4.01)
sulphates = st.number_input("sulphates", min_value=0.33, max_value=2.0)
alcohol = st.number_input("alcohol", min_value=8.4, max_value=14.9)

row = {
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol}

try:
    with open("modelo.sav", "rb") as f:
        modelo = load(f)
except FileNotFoundError:
    with open("src/modelo.sav", "rb") as f:
        modelo = load(f)

# Crear DataFrame con los datos ingresados
input_df = pd.DataFrame([row])

# Realizar la predicción cuando el usuario lo solicite
if st.button("Predecir calidad"):
    pred = modelo.predict(input_df)
    if pred[0]==0:
        st.write("No recomendable")
    else:
        st.write("Recomendable")

