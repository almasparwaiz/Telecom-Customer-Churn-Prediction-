# ================================================================
# TELECOM CUSTOMER CHURN PREDICTOR - FIXED & OPTIMIZED
# Corrected Area code type handling + robust preprocessing
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [base_dir, "", r"D:\Telecom churn prediction app\backend"]
    
    for p in paths:
        model_path = os.path.join(p, "churn_prediction_stacking_classifier.joblib")
        scaler_path = os.path.join(p, "scaler.joblib")
        features_path = os.path.join(p, "feature_columns.joblib")
        
        if all(os.path.exists(x) for x in [model_path, scaler_path, features_path]):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                feature_columns = joblib.load(features_path)
                return model, scaler, feature_columns, False
            except Exception as e:
                st.warning(f"Loading error from {p}: {e}")
                continue
                
    st.error("❌ Model files not found. Please place the three .joblib files in the app folder.")
    st.stop()

model, scaler, feature_columns, IS_DEMO = load_artifacts()

# ====================== FEATURE ENGINEERING ======================
def create_powerful_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Total_Charge"] = (d["Total day charge"] + d["Total eve charge"] +
                         d["Total night charge"] + d["Total intl charge"])
    d["Total_Minutes"] = (d["Total day minutes"] + d["Total eve minutes"] +
                          d["Total night minutes"] + d["Total intl minutes"])
    d["Total_Calls"] = (d["Total day calls"] + d["Total eve calls"] +
                        d["Total night calls"] + d["Total intl calls"])

    d["Avg_Charge_Per_Minute"] = np.where(d["Total_Minutes"] > 0,
                                          d["Total_Charge"] / d["Total_Minutes"], 0)

    d["Tenure_Group_Numeric"] = pd.qcut(d["Account length"], q=4, labels=False, duplicates="drop") if d["Account length"].nunique() > 1 else 0

    d["Voicemail_Per_Tenure"] = np.where(d["Account length"] > 0,
                                         d["Number vmail messages"] / d["Account length"], 0)
    d["Customer_Service_Calls_Per_Tenure"] = np.where(d["Account length"] > 0,
                                                      d["Customer service calls"] / d["Account length"], 0)

    if "International plan_Yes" in d.columns:
        d["Intl_Plan_and_Usage"] = d["International plan_Yes"] * d["Total intl minutes"]
    else:
        d["Intl_Plan_and_Usage"] = 0

    total_min = d["Total_Minutes"].replace(0, 1)
    d["Day_Usage_Ratio"] = d["Total day minutes"] / total_min
    d["Eve_Usage_Ratio"] = d["Total eve minutes"] / total_min
    d["Night_Usage_Ratio"] = d["Total night minutes"] / total_min
    d["Intl_Usage_Ratio"] = d["Total intl minutes"] / total_min

    return d

# ====================== PREPROCESSING (Fixed) ======================
CATEGORICAL_COLS = ["International plan", "Voice mail plan", "Area code"]
NUMERICAL_COLS = [
    "Account length", "Number vmail messages", "Total day minutes", "Total day calls", "Total day charge",
    "Total eve minutes", "Total eve calls", "Total eve charge", "Total night minutes", "Total night calls",
    "Total night charge", "Total intl minutes", "Total intl calls", "Total intl charge", "Customer service calls"
]

def preprocess(df_raw):
    # Convert input to DataFrame safely
    if isinstance(df_raw, dict):
        df = pd.DataFrame([df_raw])
    else:
        df = df_raw.copy()

    # Fix: Ensure Area code is always treated as string
    df["Area code"] = df["Area code"].astype(str)

    # One-hot encoding
    df_ohe = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Ensure expected OHE columns
    for col in ["International plan_Yes", "Voice mail plan_Yes", "Area code_415", "Area code_510"]:
        if col not in df_ohe.columns:
            df_ohe[col] = 0

    # Scale numerical columns
    scale_cols = [c for c in NUMERICAL_COLS if c in df_ohe.columns]
    df_ohe[scale_cols] = scaler.transform(df_ohe[scale_cols])

    # Feature engineering
    df_fe = create_powerful_features(df_ohe)

    # Align to training columns
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
st.caption("High-accuracy churn prediction using your Stacking Classifier")

if IS_DEMO:
    st.warning("⚠️ Demo mode active. Place your .joblib files for best accuracy.")

if st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer..."):
        X = preprocess(input_data)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        risk = "HIGH" if proba >= 0.70 else "MEDIUM" if proba >= 0.40 else "LOW"

        if prediction == 1:
            st.markdown(f'<div class="badge-churn">⚠️ HIGH RISK - Customer is likely to churn ({proba:.1%})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-stay">✅ LOW RISK - Customer is likely to stay ({proba:.1%} churn probability)</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{proba:.1%}")
        col2.metric("Retention Probability", f"{1-proba:.1%}")
        col3.metric("Risk Level", risk)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Churn Risk (%)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#fc8181" if risk == "HIGH" else "#68d391"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# ====================== BATCH PREDICTION ======================
st.markdown("---")
st.subheader("📂 Batch Prediction")

mode = st.radio("Input Mode", ["Upload CSV", "Use Local Dataset"])

if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
else:
    if st.button("Load Local Dataset"):
        try:
            path = r"D:\Telecom churn prediction app\backend\churn-bigml-80.csv"
            batch_df = pd.read_csv(path)
        except:
            st.error("Local dataset not found.")

if 'batch_df' in locals():
    try:
        with st.spinner("Running batch predictions..."):
            X_batch = preprocess(batch_df)
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:, 1]

            batch_df["Churn_Prediction"] = preds
            batch_df["Churn_Probability"] = (probs * 100).round(2)
            batch_df["Risk_Level"] = pd.cut(probs, bins=[0, 0.4, 0.7, 1], labels=["Low", "Medium", "High"])

        st.dataframe(batch_df, use_container_width=True)

        fig = px.histogram(batch_df, x="Churn_Probability", nbins=30, title="Churn Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)

        csv = batch_df.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions", csv, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Batch error: {str(e)}")

st.caption("✅ Fixed & Optimized for correct predictions • Built from your notebook")
