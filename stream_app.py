import streamlit as st

st.title("My First Streamlit App")
st.write("Hello, welcome to my Streamlit app!")

df = pd.read_csv("Synthetic-Infant-Health-Data.csv")

import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Synthetic-Infant-Health-Data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv('Synthetic-Infant-Health-Data.csv')
    st.write(df)
