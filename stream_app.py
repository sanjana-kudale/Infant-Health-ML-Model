import streamlit as st
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import IsolationForest

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

    # Debugging: Check column names after cleaning
    st.write("DataFrame columns after cleanup:", df.columns.tolist())

    # Label Encoding (if necessary)
    le = preprocessing.LabelEncoder()
    df = df.apply(lambda col: le.fit_transform(col) if col.dtype == "object" else col)

    # Outlier Detection (Optional)
    iso = IsolationForest(contamination=0.05, random_state=0)
    clean = iso.fit_predict(df)
    df = df[clean == 1]  # Remove outliers

    # Feature Selection (Ensure consistent features)
    skf = Sel
