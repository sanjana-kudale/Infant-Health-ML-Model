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
    # Check if the model has the 'feature_names_in_' attribute
    if hasattr(rf, 'feature_names_in_'):
        print("Model has 'feature_names_in_' attribute")
    else:
        print("Model does NOT have 'feature_names_in_' attribute")

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

    # 🔹 1. Apply Label Encoding (Same as Training)
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

    # 🔹 4. Ensure Correct Column Order
    missing_cols = set(feature_names) - set(rf.columns)
    extra_cols = set(rf.columns) - set(feature_names)

    # Debugging Feature Names
    st.write("Model Trained on Features:", feature_names)
    st.write("Uploaded CSV Features (After Processing):", list(df.columns))

   # 1. Ensure all required columns are present
   missing_cols = set(rf.feature_names_in_) - set(rf.columns)
if missing_cols:
    for col in missing_cols:
        df[col] = 0  # Add missing columns with default value (e.g., 0)
    
# 2. Reorder columns to match the model's training order
df = df[rf.feature_names_in_]  # This should now work without error

# 3. Ensure there are no extra columns in df
extra_cols = set(df.columns) - set(rf.feature_names_in_)
if extra_cols:
    df = df.drop(columns=extra_cols)  # Remove extra columns

# Now you can make predictions
predictions = rf.predict(df)
df["Prediction"] = predictions

# Display predictions
st.write("Predictions:", df)

# Optionally, allow for downloading predictions
st.download_button(
    label="Download Predictions",
    data=df.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv"
)

