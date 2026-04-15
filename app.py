# ================================================================
# BEST TELECOM CUSTOMER CHURN PREDICTION STREAMLIT APP
# Optimized for your notebook (Stacking Classifier + Feature Engineering)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main {background-color: #0f1117;}
    .metric-card {background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 15px;}
    .badge-churn {background: #3d1a1a; color: #fc8181; border: 1px solid #e53e3e; border-radius: 10px; padding: 18px; font-size: 1.1rem; font-weight: 600;}
    .badge-stay {background: #1a3d2a; color: #68d391; border: 1px solid #38a169; border-radius: 10px; padding: 18px; font-size: 1.1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# ====================== FEATURE ENGINEERING ======================
def create_powerful_features(df):
    d = df.copy()
    
    d["Total_Charge"] = d["Total day charge"] + d["Total eve charge"] + d["Total night charge"] + d["Total intl charge"]
    d["Total_Minutes"] = d["Total day minutes"] + d["Total eve minutes"] + d["Total night minutes"] + d["Total intl minutes"]
    d["Total_Calls"] = d["Total day calls"] + d["Total eve calls"] + d["Total night calls"] + d["Total intl calls"]
    
    d["Avg_Charge_Per_Minute"] = np.where(d["Total_Minutes"] > 0, d["Total_Charge"] / d["Total_Minutes"], 0)
    
    d["Tenure_Group_Numeric"] = pd.qcut(d["Account length"], q=4, labels=False, duplicates="drop")
    
    d["Voicemail_Per_Tenure"] = np.where(d["Account length"] > 0, d["Number vmail messages"] / d["Account length"], 0)
    d["Customer_Service_Calls_Per_Tenure"] = np.where(d["Account length"] > 0, d["Customer service calls"] / d["Account length"], 0)
    
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

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        base_dir,
        r"D:\Telecom churn prediction app\backend",
        ""
    ]
    
    for p in paths:
        model_path = os.path.join(p, "churn_prediction_stacking_classifier.joblib")
        scaler_path = os.path.join(p, "scaler.joblib")
        features_path = os.path.join(p, "feature_columns.joblib")
        
        if all(os.path.exists(x) for x in [model_path, scaler_path, features_path]):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                feature_cols = joblib.load(features_path)
                return model, scaler, feature_cols, False  # Not demo
            except:
                continue
                
    st.warning("⚠️ Model files not found. Using fallback demo mode.")
    # Simple fallback model (you can replace with full demo training if needed)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    feature_cols = []
    return model, scaler, feature_cols, True

model, scaler, feature_columns, is_demo = load_model()

# ====================== PREPROCESSING ======================
def preprocess_input(df_input):
    df = df_input.copy()
    df["Area code"] = df["Area code"].astype(str)
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=["International plan", "Voice mail plan", "Area code"], drop_first=True)
    
    # Ensure all expected columns exist
    expected_ohe = ["International plan_Yes", "Voice mail plan_Yes", "Area code_415", "Area code_510"]
    for col in expected_ohe:
        if col not in df.columns:
            df[col] = 0
    
    # Scale numerical features
    num_cols = [col for col in [
        "Account length", "Number vmail messages", "Total day minutes", "Total day calls", "Total day charge",
        "Total eve minutes", "Total eve calls", "Total eve charge", "Total night minutes", "Total night calls",
        "Total night charge", "Total intl minutes", "Total intl calls", "Total intl charge", "Customer service calls"
    ] if col in df.columns]
    
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Feature Engineering
    df_fe = create_powerful_features(df)
    
    # Align with training columns
    final_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in df_fe.columns:
        if col in final_df.columns:
            final_df[col] = df_fe[col].values[0]
    
    return final_df[feature_columns]

# ====================== SIDEBAR ======================
st.sidebar.header("📋 Customer Information")

account_length = st.sidebar.slider("Account Length (days)", 1, 250, 120)
area_code = st.sidebar.selectbox("Area Code", [408, 415, 510])
intl_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
vm_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"])
vm_messages = st.sidebar.slider("Voicemail Messages", 0, 60, 20)
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

# ====================== INPUT DATAFRAME ======================
input_dict = {
    "Account length": [account_length],
    "Area code": [area_code],
    "International plan": [intl_plan],
    "Voice mail plan": [vm_plan],
    "Number vmail messages": [vm_messages],
    "Total day minutes": [day_min],
    "Total day calls": [day_calls],
    "Total day charge": [day_charge],
    "Total eve minutes": [eve_min],
    "Total eve calls": [eve_calls],
    "Total eve charge": [eve_charge],
    "Total night minutes": [night_min],
    "Total night calls": [night_calls],
    "Total night charge": [night_charge],
    "Total intl minutes": [intl_min],
    "Total intl calls": [intl_calls],
    "Total intl charge": [intl_charge],
    "Customer service calls": [cs_calls]
}

input_df = pd.DataFrame(input_dict)

# ====================== MAIN APP ======================
st.title("📞 Telecom Customer Churn Predictor")
st.caption("Accurate churn prediction using trained Stacking Classifier")

if is_demo:
    st.warning("⚠️ Running in Demo Mode – Place your .joblib files for best results")

if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Processing..."):
        X = preprocess_input(input_df)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
        color = "#fc8181" if risk == "High" else "#fbbf24" if risk == "Medium" else "#4ade80"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.markdown(f'<div class="badge-churn">⚠️ HIGH RISK - Customer is likely to churn ({probability:.1%})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-stay">✅ LOW RISK - Customer is likely to stay ({probability:.1%} churn probability)</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Churn Probability", f"{probability:.1%}", delta=f"{risk} Risk")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Churn Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#1f2a1f"},
                    {"range": [40, 70], "color": "#2a2a1f"},
                    {"range": [70, 100], "color": "#2a1f1f"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# ====================== BATCH PREDICTION ======================
st.markdown("---")
st.subheader("📊 Batch Prediction")

tab1, tab2 = st.tabs(["Upload CSV", "Use Local Dataset"])

with tab1:
    uploaded_file = st.file_uploader("Upload your customer data (CSV)", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(batch_df)} records")

with tab2:
    local_option = st.selectbox("Select Dataset", ["Train Set (80%)", "Test Set (20%)"])
    if local_option == "Train Set (80%)":
        path = r"D:\Telecom churn prediction app\backend\churn-bigml-80.csv"
    else:
        path = r"D:\Telecom churn prediction app\backend\churn-bigml-20.csv"
    
    if st.button("Load & Predict Local Dataset"):
        batch_df = pd.read_csv(path)
        st.success(f"Loaded {len(batch_df)} records from {local_option}")

if 'batch_df' in locals():
    try:
        with st.spinner("Running batch predictions..."):
            X_batch = pd.concat([preprocess_input(pd.DataFrame([row])) for _, row in batch_df.iterrows()], ignore_index=True)
            batch_preds = model.predict(X_batch)
            batch_probs = model.predict_proba(X_batch)[:, 1]
        
        batch_df["Predicted_Churn"] = batch_preds
        batch_df["Churn_Probability"] = (batch_probs * 100).round(2)
        
        st.dataframe(batch_df, use_container_width=True)
        
        fig_hist = px.histogram(batch_df, x="Churn_Probability", nbins=30, title="Churn Probability Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        csv = batch_df.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions", csv, "churn_predictions.csv", "text/csv")
        
    except Exception as e:
        st.error(f"Error during batch prediction: {str(e)}")

st.caption("Built with Streamlit | Machine Learning | Professional Portfolio App")
