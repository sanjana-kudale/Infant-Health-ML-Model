import streamlit as st
import streamlit as st
import pandas as pd
import joblib

st.title("My First Streamlit App")
st.write("Hello, welcome to my Streamlit app!")

df = pd.read_csv("Synthetic-Infant-Health-Data.csv")

uploaded_file = st.file_uploader("rf_classifier.ipynb")

if uploaded_file is not None:
    df = pd.read_csv('Synthetic-Infant-Health-Data.csv')
    st.write(df)

grid = joblib.load("rf_classifier.pkl")
grid.best_estimator_
model = joblib.load("rf_classifier.pkl")  # Ensure 'model.pkl' is in the repo

st.write("Model loaded successfully!")

st.title("Machine Learning Model App")

params = {
    'random_state': [0, 42],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [3, 5, 10],  # Added max_depth for better tuning
    'min_samples_split': [2, 5, 10]  # Optional but useful
}

if uploaded_file is not None:
    df = pd.read_csv('Synthetic-Infant-Health-Data.csv')
    st.write("Dataset Preview:", df.head())

    # Make Predictions
    if st.button("Predict"):
        predictions = model.predict(y_pred)
        st.write("Predictions:", predictions)
