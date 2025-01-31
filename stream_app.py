import streamlit as st
import pandas as pd
import joblib

# Streamlit App Title
st.title("Infant Health Prediction App")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("rf_classifier.pkl")  # Load saved model

model = load_model()

# File Uploader
uploaded_file = st.file_uploader("Synthetic-Infant-Health-Data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())  # Show first few rows

    # Ensure the uploaded data has the same features as the trained model
    try:
        predictions = model.predict(df)  # Make Predictions
        df["Prediction"] = predictions  # Add Predictions to DataFrame
        st.write("Predictions:", df)  # Display results

        # Button to Download Results
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error in prediction: {e}")
