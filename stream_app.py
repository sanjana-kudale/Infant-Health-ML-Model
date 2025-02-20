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
    return x, y

def train_model(x, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x, y)
    joblib.dump(model, "rf_model.pkl")
    return model

def load_model():
    return joblib.load("rf_model.pkl")

st.title("Infant Health Prediction App")

if "model" not in st.session_state:
    df = load_data()
    x, y = preprocess_data(df)
    st.session_state.model = train_model(x, y)

model = st.session_state.model

uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
if uploaded_file:
    user_data = pd.read_csv(uploaded_file)
    user_data = pd.get_dummies(user_data, drop_first=True)
    prediction = model.predict(user_data)
    st.write("Predictions:", prediction)
