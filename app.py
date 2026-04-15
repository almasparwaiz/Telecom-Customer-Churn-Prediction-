# ================================================================
# TELECOM CUSTOMER CHURN PREDICTOR – CLEAN & PROFESSIONAL
# Only File Upload for Batch + Single Prediction with Gauge
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #e2e8f0; }
    .badge-churn { background: #3d1a1a; color: #fc8181; border: 1px solid #e53e3e; border-radius: 10px; padding: 20px; font-size: 1.3rem; font-weight: 700; }
    .badge-stay { background: #1a3d2a; color: #68d391; border: 1px solid #38a169; border-radius: 10px; padding: 20px; font-size: 1.3rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ====================== LOAD ARTIFACTS ======================

@st.cache_resource
def load_artifacts(): 

    
    model_path = os.path.join("churn_prediction_stacking_classifier.joblib")
    scaler_path = os.path.join("scaler.joblib")
    features_path = os.path.join("feature_columns.joblib")

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(features_path)
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Model files not found in: {base_dir}\nError: {e}")
        st.stop()

model, scaler, feature_columns = load_artifacts()

# ====================== FEATURE ENGINEERING ======================
def create_powerful_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Total_Charge"] = (d.get("Total day charge", 0) + d.get("Total eve charge", 0) +
                         d.get("Total night charge", 0) + d.get("Total intl charge", 0))
    d["Total_Minutes"] = (d.get("Total day minutes", 0) + d.get("Total eve minutes", 0) +
                          d.get("Total night minutes", 0) + d.get("Total intl minutes", 0))
    
    d["Avg_Charge_Per_Minute"] = np.where(d["Total_Minutes"] > 0, d["Total_Charge"] / d["Total_Minutes"], 0)
    d["Voicemail_Per_Tenure"] = np.where(d["Account length"] > 0, d.get("Number vmail messages", 0) / d["Account length"], 0)
    d["Customer_Service_Calls_Per_Tenure"] = np.where(d["Account length"] > 0, d.get("Customer service calls", 0) / d["Account length"], 0)

    total_min = d["Total_Minutes"].replace(0, 1)
    d["Day_Usage_Ratio"] = d.get("Total day minutes", 0) / total_min
    d["Eve_Usage_Ratio"] = d.get("Total eve minutes", 0) / total_min
    d["Night_Usage_Ratio"] = d.get("Total night minutes", 0) / total_min
    d["Intl_Usage_Ratio"] = d.get("Total intl minutes", 0) / total_min

    return d

# ====================== PREPROCESSING ======================
CATEGORICAL_COLS = ["International plan", "Voice mail plan", "Area code"]
NUMERICAL_COLS = [
    "Account length", "Number vmail messages", "Total day minutes", "Total day calls", "Total day charge",
    "Total eve minutes", "Total eve calls", "Total eve charge", "Total night minutes", "Total night calls",
    "Total night charge", "Total intl minutes", "Total intl calls", "Total intl charge", "Customer service calls"
]

def preprocess(df_raw):
    if isinstance(df_raw, dict):
        df = pd.DataFrame([df_raw])
    else:
        df = df_raw.copy()

    df["Area code"] = df["Area code"].astype(str)

    df_ohe = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Ensure dummy columns
    for col in ["International plan_Yes", "Voice mail plan_Yes", "Area code_415", "Area code_510"]:
        if col not in df_ohe.columns:
            df_ohe[col] = 0

    # Scaling
    scale_cols = [c for c in NUMERICAL_COLS if c in df_ohe.columns]
    if scale_cols:
        df_ohe[scale_cols] = scaler.transform(df_ohe[scale_cols])

    # Feature Engineering
    df_fe = create_powerful_features(df_ohe)

    # Align columns
    final = pd.DataFrame(0, index=df_fe.index, columns=feature_columns)
    for col in df_fe.columns:
        if col in final.columns:
            final[col] = df_fe[col]

    return final[feature_columns]

# ====================== SIDEBAR ======================
st.sidebar.header("📋 Customer Profile")

account_length = st.sidebar.slider("Account Length (days)", 1, 250, 120)
area_code = st.sidebar.selectbox("Area Code", [408, 415, 510])
intl_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
vm_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"])
vm_messages = st.sidebar.slider("Voicemail Messages", 0, 60, 15)
cs_calls = st.sidebar.slider("Customer Service Calls", 0, 9, 2)

day_min = st.sidebar.slider("Day Minutes", 0.0, 400.0, 180.0, 0.5)
day_calls = st.sidebar.slider("Day Calls", 0, 200, 110)
day_charge = st.sidebar.slider("Day Charge ($)", 0.0, 70.0, 30.6, 0.1)

eve_min = st.sidebar.slider("Evening Minutes", 0.0, 400.0, 200.0, 0.5)
eve_calls = st.sidebar.slider("Evening Calls", 0, 200, 100)
eve_charge = st.sidebar.slider("Evening Charge ($)", 0.0, 35.0, 17.0, 0.1)

night_min = st.sidebar.slider("Night Minutes", 0.0, 400.0, 190.0, 0.5)
night_calls = st.sidebar.slider("Night Calls", 0, 200, 95)
night_charge = st.sidebar.slider("Night Charge ($)", 0.0, 25.0, 8.6, 0.1)

intl_min = st.sidebar.slider("International Minutes", 0.0, 30.0, 10.0, 0.1)
intl_calls = st.sidebar.slider("International Calls", 0, 20, 4)
intl_charge = st.sidebar.slider("International Charge ($)", 0.0, 10.0, 2.7, 0.1)

input_data = {
    "Account length": account_length,
    "Area code": area_code,
    "International plan": intl_plan,
    "Voice mail plan": vm_plan,
    "Number vmail messages": vm_messages,
    "Total day minutes": day_min,
    "Total day calls": day_calls,
    "Total day charge": day_charge,
    "Total eve minutes": eve_min,
    "Total eve calls": eve_calls,
    "Total eve charge": eve_charge,
    "Total night minutes": night_min,
    "Total night calls": night_calls,
    "Total night charge": night_charge,
    "Total intl minutes": intl_min,
    "Total intl calls": intl_calls,
    "Total intl charge": intl_charge,
    "Customer service calls": cs_calls,
}

# ====================== MAIN APP ======================
st.title("📞 Telecom Customer Churn Predictor")
st.caption("High-accuracy Stacking Classifier")

if st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer..."):
        X = preprocess(input_data)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        if prediction == 1:
            st.markdown(f'<div class="badge-churn">⚠️ HIGH RISK - Customer is likely to churn ({proba:.1%})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-stay">✅ LOW RISK - Customer is likely to stay ({proba:.1%} churn probability)</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{proba:.1%}")
        col2.metric("Retention Probability", f"{1-proba:.1%}")
        col3.metric("Risk Level", "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.4 else "LOW")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Churn Risk (%)"},
            gauge={"axis": {"range": [0, 100]}, 
                   "bar": {"color": "#fc8181" if proba >= 0.7 else "#fbbf24" if proba >= 0.4 else "#68d391"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# ====================== BATCH PREDICTION ======================
st.markdown("---")
st.subheader("📂 Batch Prediction (File Upload)")

uploaded = st.file_uploader("Upload customer data CSV", type=["csv"])

if uploaded:
    try:
        batch_df = pd.read_csv(uploaded)
        with st.spinner("Running batch predictions..."):
            X_batch = preprocess(batch_df)
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:, 1]

            batch_df["Churn_Prediction"] = preds
            batch_df["Churn_Probability (%)"] = (probs * 100).round(2)
            batch_df["Risk_Level"] = pd.cut(probs, bins=[0, 0.4, 0.7, 1], labels=["Low", "Medium", "High"])

        st.dataframe(batch_df, use_container_width=True)

        fig = px.histogram(batch_df, x="Churn_Probability (%)", nbins=30, title="Churn Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)

        csv = batch_df.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

st.caption("✅ Clean Professional Telecom Customer Churn Prediction App")
