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

        # 🔹 Check the first few rows to ensure the data is loaded correctly
    st.write("Preview of Uploaded File:", df.head())

    # 🔹 Check the data types of each column
    st.write("Data types of each column:", df.dtypes)

    # 🔹 Check for any missing values in the dataset
    st.write("Check for any missing values in the dataset:", df.isnull().sum())

    # 🔹 Remove rows with all zeros or NaN values
    df = df.loc[(df != 0).any(axis=1)]  # Remove rows where all columns are zero
    st.write("Shape after removing empty rows:", df.shape)

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

    # 🔹 Keep Only Numeric Columns, Replace NaN with 0
    df = df.select_dtypes(include=[np.number])  # Keep only numeric data
    df = df.fillna(0)  # Replace NaN with 0
    df = df.astype(float)  # Convert all columns to float

        # 🔹 Check for columns with all zeros or empty values
    st.write("Check for columns with all zeros or empty values:")
    st.write(df.columns[(df == 0).all()])  # Columns filled with zeros
    st.write(df.columns[df.isnull().all()])  # Columns with all NaN values


    # 🔹 Preview the data after preprocessing
    st.write("Data after simple preprocessing (before Isolation Forest):", df.head())


    # 🔹 Check the data after preprocessing
    st.write("Data after preprocessing (before Isolation Forest):", df.head())

    # 🔹 🚀 FIX: Ensure df is not empty before applying Isolation Forest
    # 🔹 Ensure the DataFrame is not empty before making predictions
    if df.empty:
        st.error("Error: Processed DataFrame is empty. Please check the uploaded file.")
        st.stop()

    # 🔹 Apply Isolation Forest for Outlier Detection (Skip for debugging)
    
    # 🔹 Apply Isolation Forest for Outlier Detection
    # 🔹 Apply Isolation Forest for Outlier Detection
    try:
        st.write(f"Shape of data before Isolation Forest: {df.shape}")
        # Skip outlier removal for debugging
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


    # Add missing columns with 0 values
    for col in missing_cols:
        df[col] = 0

    df = df[feature_names]  # Reorder columns to match training
    
    # Debugging: Check the features before making predictions
    st.write("Model Trained on Features:", feature_names)
    st.write("Processed Features from Uploaded CSV:", list(df.columns))


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
