import streamlit as st
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

st.title("Infant Health Prediction App")

# Load Model & Feature Names
@st.cache_resource
def load_model():
    return joblib.load("rf_classifier.pkl")  # Load trained model

@st.cache_resource
def load_features():
    return joblib.load("feature_names.pkl")  # Load selected feature names

rf = load_model()
feature_names = load_features()

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 🔹 Remove "Unnamed: 0" if it exists (this is usually an index column)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 🔹 Preview the first few rows
    st.write("Preview of Uploaded File:", df.head())

    # 🔹 Handle Object Columns (Convert to Numeric)
    for col in df.select_dtypes(include=["object"]).columns:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert categories to numbers

    # 🔹 Keep Only Numeric Columns and Fill NaN with 0
    df = df.fillna(0)  # Replace NaN with 0
    df = df.astype(float)  # Convert all columns to float

    # 🔹 Ensure All Feature Names Match with Training Data
    missing_cols = set(feature_names) - set(df.columns)
    extra_cols = set(df.columns) - set(feature_names)

    # Add missing columns with 0 values (including XrayReport_Oligaemic if it's missing)
    for col in missing_cols:
        df[col] = 0

    # 🔹 Log which columns were added or removed
    st.write("Missing columns added:", missing_cols)
    st.write("Extra columns removed:", extra_cols)

    # Remove extra columns that are not part of the training data
    df = df[feature_names]  # Reorder columns to match training

    # 🔹 Ensure the DataFrame is not empty before making predictions
    if df.empty:
        st.error("Error: Processed DataFrame is empty. Please check the uploaded file.")
        st.stop()

    # 🔹 Ensure that the uploaded data has the same columns as the training data
    st.write("Processed Data (Before Prediction):", df.head())  # See the final processed data
    st.write("Expected Features:", feature_names)

    try:
        # 🔹 Make Predictions
        predictions = rf.predict(df)
        df["Prediction"] = predictions
        st.write("Predictions:", df)

        # 🔹 Download Predictions
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Prediction Error: {e}")