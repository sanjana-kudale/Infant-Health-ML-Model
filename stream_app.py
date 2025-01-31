import streamlit as st
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, chi2

st.title("Infant Health Prediction App")

# Load Model & Feature Names
@st.cache_resource
def load_model():
    return joblib.load("rf_classifier.pkl")  # Load trained model

@st.cache_resource
def load_features():
    return joblib.load("feature_names.pkl")  # Load selected feature names

# Load the model and feature names
rf = load_model()
feature_names = load_features()

# File Uploader
uploaded_file = st.file_uploader("Synthetic-Infant-Health-Data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())  # Display first few rows

    # Remove "Unnamed: 0" if it exists in the dataset (often an extra index column)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Debugging: Check columns after cleanup
    st.write("DataFrame columns after cleanup:", df.columns.tolist())

    # Ensure that all the required columns are present (like `XrayReport_Oligaemic`)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0

    # Debugging: Check columns after ensuring all features are present
    st.write("DataFrame columns after adding missing features:", df.columns.tolist())

    # Reorder columns to match the model's expected feature names
    df = df[feature_names]

    # Debugging: Check columns after reordering
    st.write("DataFrame columns after reordering:", df.columns.tolist())

    # Try making predictions
    try:
        predictions = rf.predict(df)
        df["Prediction"] = predictions

        # Display predictions
        st.write("Predictions:", df)

        # Allow users to download the predictions as CSV
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error in prediction: {e}")
