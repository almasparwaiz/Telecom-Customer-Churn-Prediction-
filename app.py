# =========================
# 📦 IMPORT LIBRARIES
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier, VotingClassifier

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# =========================
# 🎯 HEADER
# =========================
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🚀 AI-Powered Customer Churn Prediction System
</h1>
<p style='text-align: center;'>
Predict customer churn with high accuracy using Machine Learning
</p>
""", unsafe_allow_html=True)

# =========================
# 📂 LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("churn_prediction_stacking_classifier.joblib")
        scaler = joblib.load("scaler.joblib")
        features = joblib.load("feature_columns.joblib")
        return model, scaler, features
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        st.stop()

model, scaler, feature_columns = load_artifacts()

# =========================
# 🧠 FEATURE ENGINEERING
# =========================
def create_features(df):
    df = df.copy()

    # Safe column handling (fix KeyError)
    charge_cols = [c for c in ['Total day charge','Total eve charge','Total night charge','Total intl charge'] if c in df.columns]
    minute_cols = [c for c in ['Total day minutes','Total eve minutes','Total night minutes','Total intl minutes'] if c in df.columns]

    df['Total_Charge'] = df[charge_cols].sum(axis=1) if charge_cols else 0
    df['Total_Minutes'] = df[minute_cols].sum(axis=1) if minute_cols else 0

    df['Avg_Charge_Per_Minute'] = df['Total_Charge'] / (df['Total_Minutes'] + 1)

    df['Customer_Service_Intensity'] = df['Customer service calls'] / (df['Account length'] + 1)

    return df

# =========================
# 🧾 INPUT UI
# =========================
st.sidebar.header("📥 Customer Input")

def user_input():
    data = {
        'Account length': st.sidebar.slider("Account Length", 1, 300, 100),
        'Number vmail messages': st.sidebar.slider("Voicemail Messages", 0, 50, 5),
        'Total day minutes': st.sidebar.slider("Day Minutes", 0, 400, 200),
        'Total eve minutes': st.sidebar.slider("Evening Minutes", 0, 400, 200),
        'Total night minutes': st.sidebar.slider("Night Minutes", 0, 400, 200),
        'Total intl minutes': st.sidebar.slider("International Minutes", 0, 50, 10),
        'Customer service calls': st.sidebar.slider("Customer Service Calls", 0, 10, 1),
        'International plan_Yes': st.sidebar.selectbox("International Plan", [0,1]),
        'Voice mail plan_Yes': st.sidebar.selectbox("Voicemail Plan", [0,1])
    }
    return pd.DataFrame([data])

input_df = user_input()

# =========================
# 🔄 PREPROCESSING
# =========================
def preprocess(df):
    df = create_features(df)

    # Ensure all columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # Scale safely
    try:
        df_scaled = scaler.transform(df)
        return pd.DataFrame(df_scaled, columns=feature_columns)
    except Exception:
        return df

processed = preprocess(input_df)

# =========================
# 🔮 PREDICTION
# =========================
st.subheader("📊 Prediction Result")

if st.button("🚀 Predict Churn"):

    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("⚠️ Customer WILL CHURN")
        else:
            st.success("✅ Customer will STAY")

    with col2:
        st.metric("Churn Probability", f"{prob:.2%}")

    # =========================

# =========================
# 📌 FOOTER
# =========================
st.markdown("""
---
💼 **Developed for Portfolio & Client Projects**  
📧 Contact: almasparwaiz1@gmail.com  
""")
