import streamlit as st

st.title("Upload Your Jupyter Notebook")

uploaded_file = st.file_uploader("Upload your rf_classifier.ipynb file", type="ipynb")

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    with open("rf_classifier.ipynb", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info("File saved as rf_classifier.ipynb")
