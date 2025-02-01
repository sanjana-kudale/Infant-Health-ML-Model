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

    # 🔹 Drop "Unnamed: 0" column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    st.write("Uploaded Data (Before Processing):", df.head())  # Show raw data

    # 🔹 Convert categorical variables to numerical (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)

    # 🔹 Fix Label Encoding for Categorical Columns
    for col in df.select_dtypes(include=["object"]).columns:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert categories to numbers

    # 🔹 Keep Only Numeric Columns
    df = df.select_dtypes(include=[np.number])  # Keep only numeric data
    df = df.fillna(0)  # Replace NaN with 0
    df = df.astype(float)  # Convert all columns to float

    # 🔹 🚀 FIX: Ensure df is not empty before applying Isolation Forest
    if df.empty:
        st.error("Error: Processed DataFrame is empty. Please check the uploaded file.")
        st.stop()  # Stop execution if df is empty

    
    # 🔹 Apply Isolation Forest for Outlier Detection
    # 🔹 Apply Isolation Forest for Outlier Detection
    try:
        st.write(f"Shape of data before Isolation Forest: {df.shape}")
        st.write("Data after preprocessing (before Isolation Forest):", df.head())  # Check data
        iso = IsolationForest(contamination=0.01, random_state=0)  # Use a lower contamination rate
        clean = iso.fit_predict(df)
        df = df[clean == 1]  # Remove outliers
    except Exception as e:
        st.error(f"Isolation Forest Error: {e}")
        st.stop()


    # 🔹 Ensure All Feature Names Match
    missing_cols = set(feature_names) - set(df.columns)
    extra_cols = set(df.columns) - set(feature_names)

    # Add missing columns with 0 values
    for col in missing_cols:
        df[col] = 0

    df = df[feature_names]  # Reorder columns to match training

    # Debugging
    st.write("Model Trained on Features:", feature_names)
    st.write("Uploaded CSV Features (After Processing):", list(df.columns))

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
