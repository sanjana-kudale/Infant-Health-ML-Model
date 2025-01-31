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

    # Debugging: Check column names after cleanup
    st.write("DataFrame columns after cleanup:", df.columns.tolist())

    # Label Encoding (if necessary)
    le = preprocessing.LabelEncoder()
    df = df.apply(lambda col: le.fit_transform(col) if col.dtype == "object" else col)

    # Outlier Detection (Optional)
    iso = IsolationForest(contamination=0.05, random_state=0)
    clean = iso.fit_predict(df)
    df = df[clean == 1]  # Remove outliers

    # Feature Selection (Ensure consistent features)
    # Apply SelectKBest only on numerical features
    X = df.select_dtypes(include=['number'])  # Select numerical columns only

    # Apply SelectKBest with chi-squared test (requires positive values)
    if X.shape[0] > 0:  # Ensure there are rows to work with
        skf = SelectKBest(score_func=chi2, k=5)  # Select top 5 features
        X_new = skf.fit_transform(X, [0] * len(X))  # Dummy target to keep selection consistent
        df_new = pd.DataFrame(X_new, columns=feature_names)  # Assign selected features
    else:
        st.error("No numerical data available for feature selection.")
        df_new = df  # Keep original data if no numerical data exists

    # Debugging: Check columns before making predictions
    st.write("DataFrame columns after feature selection:", df_new.columns.tolist())
    st.write("Expected feature names:", feature_names)

    # Ensure the input data columns match the model's expected feature names
    missing_cols = set(feature_names) - set(df_new.columns)
    extra_cols = set(df_new.columns) - set(feature_names)

    # Add missing columns with default value 0
    for col in missing_cols:
        df_new[col] = 0

    # Reorder columns to match the training data's feature names
    df_new = df_new[feature_names]

    # Drop extra columns
    if extra_cols:
        df_new = df_new.drop(columns=extra_cols)

    # Final Debugging: Check column alignment after handling missing/extra columns
    st.write("DataFrame columns after reordering and handling missing columns:", df_new.columns.tolist())

    try:
        # Make Predictions
        predictions = rf.predict(df_new)
        df_new["Prediction"] = predictions

        # Display predictions
        st.write("Predictions:", df_new)

        # Allow users to download the predictions as CSV
        st.download_button(
            label="Download Predictions",
            data=df_new.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error in prediction: {e}")
