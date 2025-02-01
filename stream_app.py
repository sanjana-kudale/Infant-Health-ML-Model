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

    # 🔹 Preview the first few rows
    st.write("Preview of Uploaded File:", df.head())

    # 🔹 Check the data types of each column
    st.write("Data types of each column:", df.dtypes)

    # 🔹 Handle Object Columns (Convert to Numeric)
    for col in df.select_dtypes(include=["object"]).columns:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert categories to numbers

    # 🔹 Keep Only Numeric Columns and Fill NaN with 0
    df = df.fillna(0)  # Replace NaN with 0
    df = df.astype(float)  # Convert all columns to float

    # 🔹 Check if any columns are now filled with zeros or empty
    st.write("Check for columns with all zeros or empty values:")
    st.write(df.columns[(df == 0).all()])  # Columns filled with zeros
    st.write(df.columns[df.isnull().all()])  # Columns with all NaN values

    # 🔹 Remove rows with all zeros or NaN values
    df = df.loc[(df != 0).any(axis=1)]  # Remove rows where all columns are zero
    st.write("Shape after removing empty rows:", df.shape)

    # 🔹 Preview the data after preprocessing
    st.write("Data after preprocessing (before Isolation Forest):", df.head())

    # 🔹 Apply Isolation Forest for Outlier Detection (optional)
    try:
        st.write(f"Shape of data before Isolation Forest: {df.shape}")
        # iso = IsolationForest(contamination=0.01, random_state=0)
        # clean = iso.fit_predict(df)
        # df = df[clean == 1]  # Skip this step for now
    except Exception as e:
        st.error(f"Isolation Forest Error: {e}")
        st.stop()

    # 🔹 Ensure All Feature Names Match
    missing_cols = set(feature_names) - set(df.columns)
    extra_cols = set(df.columns) - set(feature_names)

    st.write("Missing columns:", missing_cols)
    st.write("Extra columns:", extra_cols)

    # 🔹 Ensure the DataFrame is not empty before making predictions
    if df.empty:
        st.error("Error: Processed DataFrame is empty. Please check the uploaded file.")
        st.stop()

    st.write("Processed Data (Before Prediction):", df.head())  # See the final processed data

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
