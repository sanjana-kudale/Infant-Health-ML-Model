import streamlit as st
import streamlit as st
import pandas as pd
import joblib

st.title("My First Streamlit App")
st.write("Hello, welcome to my Streamlit app!")

df = pd.read_csv("Synthetic-Infant-Health-Data.csv")

uploaded_file = st.file_uploader("Synthetic-Infant-Health-Data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv('Synthetic-Infant-Health-Data.csv')
    st.write(df)

grid = joblib.load("rf_classifier.pkl")
grid.best_estimator_
