import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Infant Health Prediction App")

# Load Model
model = joblib.load("rf_classifier.pkl")  # Ensure the model file is in the same directory

# File Uploader
uploaded_file = st.file_uploader("Synthetic-Infant-Health-Data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())  # Display sample of uploaded data

    # Make predictions
    predictions = model.predict(df)

    # Display results
    df["Prediction"] = predictions
    st.write("Predictions:", df)
    st.download_button(
        label="Download Predictions",
        data=df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
