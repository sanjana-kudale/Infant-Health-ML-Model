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

rf = load_model()
feature_names = load_features()

# File Uploader
uploaded_file = st.file_uploader("Synthetic-Infant-Health-Data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())  # Display first few rows

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    st.write("Uploaded Data (Before Processing):", df.head())  # Show data before processing

    # 🔹 Apply Label Encoding (Ensuring categorical data is converted)
    le = preprocessing.LabelEncoder()
    df = df.apply(lambda col: le.fit_transform(col) if col.dtype == "object" else col)

    # 🔹 2. Apply Isolation Forest for Outlier Detection
    iso = IsolationForest(contamination=0.05, random_state=0)
    clean = iso.fit_predict(df)
    df = df[clean == 1]  # Remove outliers

    # 🔹 3. Feature Selection (Ensure same top 5 features are used)
    skf = SelectKBest(k=5, score_func=chi2)
    df_new = skf.fit_transform(df, [0] * len(df))  # Dummy target to keep feature selection consistent

    # Convert back to DataFrame with correct feature names
    df = pd.DataFrame(df_new, columns=feature_names)

    st.write("🔍 Debugging Step: Checking DataFrame and Feature Names")

    # Check if df_new exists
    if 'df_new' in locals():
        st.write("✅ df_new exists. Shape:", df_new.shape)
    else:
        st.error("❌ df_new is not defined!")

    # Check if df exists
    if 'df' in locals():
        st.write("✅ df exists. Shape:", df.shape)
    else:
        st.error("❌ df is not defined!")

        # Check if feature_names exist
        st.write("Model Trained on Features:", feature_names)

        # Check if df_new has the correct columns
        st.write("df_new Columns (Before Reordering):", df_new.columns.tolist())

    
# 🔹 4. Ensure Correct Column Order
    missing_cols = set(feature_names) - set(df.columns)
    extra_cols = set(df.columns) - set(feature_names)

    # Debugging Feature Names
    st.write("Model Trained on Features:", feature_names)
    st.write("Uploaded CSV Features (After Processing):", list(df.columns))

    for col in missing_cols:
        df[col] = 0  # Add missing columns with 0 values

    # Fix for extra columns
    if extra_cols:  # Properly indented `if` block
        df_new = df_new.drop(columns=extra_cols)

    # Final Debugging: Check column alignment after handling missing/extra columns
    st.write("DataFrame columns after reordering and handling missing columns:", df_new.columns.tolist())

    df = df[feature_names]  # Reorder columns to match training

    st.write("Data ready for prediction (Feature Names):", df_final.columns.tolist())
    
    predictions = rf.predict(df_new)
    df["Prediction"] = predictions
    st.write("Predictions:", df["Prediction"].head())
