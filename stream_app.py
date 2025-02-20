import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def load_data():
    df = pd.read_csv("Synthetic-Infant-Health-Data.csv")
    df.drop(["DuctFlow", "CardiacMixing"], axis=1, inplace=True)
    return df

def preprocess_data(df):
    x = df.drop("Sick", axis=1)
    y = df["Sick"].apply(lambda x: 1 if x == "yes" else 0)
    x = pd.get_dummies(x, drop_first=True)
    return x, y, x.columns.tolist()

def train_model(x, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x, y)
    joblib.dump((model, x.columns.tolist()), "rf_model.pkl")
    return model

def load_model():
    model, feature_columns = joblib.load("rf_model.pkl")
    return model, feature_columns

st.title("Infant Health Prediction App")

if "model" not in st.session_state:
    df = load_data()
    x, y, feature_columns = preprocess_data(df)
    st.session_state.model, st.session_state.feature_columns = train_model(x, y), feature_columns

model, feature_columns = st.session_state.model, st.session_state.feature_columns

uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
if uploaded_file:
    user_data = pd.read_csv(uploaded_file)
    user_data = pd.get_dummies(user_data, drop_first=True)
    
    # Ensure uploaded data has the same columns as training data
    for col in feature_columns:
        if col not in user_data:
            user_data[col] = 0
    user_data = user_data[feature_columns]  # Ensure column order
    
    prediction = model.predict(user_data)
    st.write("Predictions:", prediction)
