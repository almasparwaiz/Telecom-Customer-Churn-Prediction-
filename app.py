# ================================================================
# TELECOM CUSTOMER CHURN PREDICTION APP
# Full functional version — predictions vary with real input data
# ================================================================
# Requirements:
#   pip install streamlit pandas numpy joblib plotly scikit-learn
#
# Run:
#   streamlit run telecom_churn_app.py
#
# Expected model files (place in same folder OR set BACKEND_DIR):
#   - churn_prediction_stacking_classifier.joblib
#   - scaler.joblib
#   - feature_columns.joblib
#
# If model files are missing, the app trains a demo model
# automatically using synthetic data so you can still run it.
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------
st.markdown("""
<style>
/* ── General ──────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.main { background: #0f1117; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* ── Metric cards ─────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 14px 18px;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111622;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: #a0aec0;
}

/* ── Section headers ──────────────────────────────────── */
.section-tag {
    display: inline-block;
    background: #1a2744;
    color: #63b3ed;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
}

/* ── Result badges ────────────────────────────────────── */
.badge-churn {
    background: #3d1a1a; color: #fc8181;
    border: 1px solid #e53e3e;
    border-radius: 8px; padding: 16px 20px;
    font-weight: 600; font-size: 1.05rem;
}
.badge-stay {
    background: #1a3d2a; color: #68d391;
    border: 1px solid #38a169;
    border-radius: 8px; padding: 16px 20px;
    font-weight: 600; font-size: 1.05rem;
}

/* ── Dataframe ────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# FEATURE ENGINEERING  (mirrors notebook exactly)
# ================================================================
def create_powerful_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates all engineered features from the training notebook.
    Input df must already have original numerics scaled AND
    OHE columns present (International plan_Yes, etc.)
    """
    d = df.copy()

    # 1. Aggregate charges / minutes / calls
    d["Total_Charge"] = (d["Total day charge"] + d["Total eve charge"]
                         + d["Total night charge"] + d["Total intl charge"])
    d["Total_Minutes"] = (d["Total day minutes"] + d["Total eve minutes"]
                          + d["Total night minutes"] + d["Total intl minutes"])
    d["Total_Calls"] = (d["Total day calls"] + d["Total eve calls"]
                        + d["Total night calls"] + d["Total intl calls"])

    # 2. Average charge per minute (avoid divide-by-zero)
    d["Avg_Charge_Per_Minute"] = np.where(
        d["Total_Minutes"] != 0,
        d["Total_Charge"] / d["Total_Minutes"], 0
    )
    d["Avg_Charge_Per_Minute"] = (d["Avg_Charge_Per_Minute"]
                                   .replace([np.inf, -np.inf], np.nan)
                                   .fillna(0))

    # 3. Tenure group (numeric)
    if d["Account length"].nunique() > 1:
        d["Tenure_Group_Numeric"] = pd.qcut(
            d["Account length"], q=4, labels=False, duplicates="drop"
        )
    else:
        d["Tenure_Group_Numeric"] = 0

    # 4. Voicemail per tenure
    d["Voicemail_Per_Tenure"] = np.where(
        d["Account length"] != 0,
        d["Number vmail messages"] / d["Account length"], 0
    )
    d["Voicemail_Per_Tenure"] = (d["Voicemail_Per_Tenure"]
                                   .replace([np.inf, -np.inf], np.nan)
                                   .fillna(0))

    # 5. Customer-service calls per tenure
    d["Customer_Service_Calls_Per_Tenure"] = np.where(
        d["Account length"] != 0,
        d["Customer service calls"] / d["Account length"], 0
    )
    d["Customer_Service_Calls_Per_Tenure"] = (
        d["Customer_Service_Calls_Per_Tenure"]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0)
    )

    # 6. International plan × intl minutes interaction
    if "International plan_Yes" in d.columns:
        d["Intl_Plan_and_Usage"] = (d["International plan_Yes"]
                                     * d["Total intl minutes"])
    else:
        d["Intl_Plan_and_Usage"] = 0

    # 7. Time-of-day usage ratios
    d["Day_Usage_Ratio"] = np.where(
        d["Total_Minutes"] != 0,
        d["Total day minutes"] / d["Total_Minutes"], 0
    )
    d["Eve_Usage_Ratio"] = np.where(
        d["Total_Minutes"] != 0,
        d["Total eve minutes"] / d["Total_Minutes"], 0
    )
    d["Night_Usage_Ratio"] = np.where(
        d["Total_Minutes"] != 0,
        d["Total night minutes"] / d["Total_Minutes"], 0
    )
    d["Intl_Usage_Ratio"] = np.where(
        d["Total_Minutes"] != 0,
        d["Total intl minutes"] / d["Total_Minutes"], 0
    )

    return d


# ================================================================
# DEMO MODEL BUILDER
# Trains a quick model in-memory when .joblib files are absent.
# ================================================================
def build_demo_model():
    """
    Generates synthetic telecom churn data and trains a
    stacking classifier so the app is always runnable.
    """
    from sklearn.ensemble import (GradientBoostingClassifier,
                                   RandomForestClassifier,
                                   StackingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n = 3_000

    # ── Synthetic raw features ──────────────────────────────────
    acct_len   = np.random.randint(1, 250, n)
    area_code  = np.random.choice([408, 415, 510], n)
    intl_plan  = np.random.choice(["No", "Yes"], n, p=[0.9, 0.1])
    vm_plan    = np.random.choice(["No", "Yes"], n, p=[0.72, 0.28])
    vmail_msgs = np.where(vm_plan == "Yes",
                          np.random.randint(0, 51, n), 0)
    day_min    = np.random.uniform(0, 400, n)
    day_calls  = np.random.randint(0, 200, n)
    day_chg    = day_min * 0.17
    eve_min    = np.random.uniform(0, 400, n)
    eve_calls  = np.random.randint(0, 200, n)
    eve_chg    = eve_min * 0.085
    night_min  = np.random.uniform(0, 400, n)
    night_calls= np.random.randint(0, 200, n)
    night_chg  = night_min * 0.045
    intl_min   = np.random.uniform(0, 30, n)
    intl_calls = np.random.randint(0, 20, n)
    intl_chg   = intl_min * 0.27
    csc        = np.random.randint(0, 10, n)

    # ── Label: churn driven by real-world signal patterns ───────
    log_odds = (
        -2.5
        + 1.8 * (csc > 3).astype(float)
        + 2.1 * (intl_plan == "Yes").astype(float)
        + 0.9 * ((day_min - 180) / 54)
        - 0.6 * (vm_plan == "Yes").astype(float)
        + 0.4 * np.random.randn(n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Account length": acct_len,
        "Area code": area_code,
        "International plan": intl_plan,
        "Voice mail plan": vm_plan,
        "Number vmail messages": vmail_msgs,
        "Total day minutes": day_min,
        "Total day calls": day_calls,
        "Total day charge": day_chg,
        "Total eve minutes": eve_min,
        "Total eve calls": eve_calls,
        "Total eve charge": eve_chg,
        "Total night minutes": night_min,
        "Total night calls": night_calls,
        "Total night charge": night_chg,
        "Total intl minutes": intl_min,
        "Total intl calls": intl_calls,
        "Total intl charge": intl_chg,
        "Customer service calls": csc,
        "Churn": churn,
    })

    # ── Preprocessing ───────────────────────────────────────────
    cat_cols = ["International plan", "Voice mail plan", "Area code"]
    df["Area code"] = df["Area code"].astype(str)
    df_ohe = pd.get_dummies(df.drop(columns=["Churn"]),
                            columns=cat_cols, drop_first=True)

    num_cols = [
        "Account length", "Number vmail messages",
        "Total day minutes", "Total day calls", "Total day charge",
        "Total eve minutes", "Total eve calls", "Total eve charge",
        "Total night minutes", "Total night calls", "Total night charge",
        "Total intl minutes", "Total intl calls", "Total intl charge",
        "Customer service calls",
    ]
    scaler = StandardScaler()
    df_ohe[num_cols] = scaler.fit_transform(df_ohe[num_cols])

    df_fe = create_powerful_features(df_ohe)
    y = df["Churn"].values

    feature_columns = list(df_fe.columns)

    # ── Stacking classifier ─────────────────────────────────────
    base_estimators = [
        ("gb",  GradientBoostingClassifier(n_estimators=80,
                                            max_depth=4,
                                            learning_rate=0.1,
                                            random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=80,
                                        max_depth=6,
                                        random_state=42)),
    ]
    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=500),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    model.fit(df_fe[feature_columns], y)

    return model, scaler, feature_columns


# ================================================================
# LOAD ARTIFACTS  (model, scaler, feature_columns)
# ================================================================
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """
    Try to load from disk; fall back to training a demo model.
    Set BACKEND_DIR env-var or edit the path below.
    """
    # ── Try real files first ────────────────────────────────────
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.environ.get("BACKEND_DIR", ""),
        r"D:\Telecom churn prediction app\backend",
    ]

    for d in search_dirs:
        if not d:
            continue
        m_path  = os.path.join(d, "churn_prediction_stacking_classifier.joblib")
        sc_path = os.path.join(d, "scaler.joblib")
        fc_path = os.path.join(d, "feature_columns.joblib")
        if all(os.path.exists(p) for p in [m_path, sc_path, fc_path]):
            try:
                import joblib
                model           = joblib.load(m_path)
                scaler          = joblib.load(sc_path)
                feature_columns = joblib.load(fc_path)
                return model, scaler, feature_columns, False   # False = not demo
            except Exception as e:
                st.warning(f"Found files in {d} but couldn't load: {e}")

    # ── Fall back to demo model ─────────────────────────────────
    model, scaler, feature_columns = build_demo_model()
    return model, scaler, feature_columns, True   # True = demo


model, scaler, feature_columns, IS_DEMO = load_artifacts()


# ================================================================
# PREPROCESSING PIPELINE  (mirrors training notebook)
# ================================================================
CATEGORICAL_COLS   = ["International plan", "Voice mail plan", "Area code"]
NUMERICAL_COLS     = [
    "Account length", "Number vmail messages",
    "Total day minutes", "Total day calls", "Total day charge",
    "Total eve minutes", "Total eve calls", "Total eve charge",
    "Total night minutes", "Total night calls", "Total night charge",
    "Total intl minutes", "Total intl calls", "Total intl charge",
    "Customer service calls",
]

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1. Convert Area code to string so get_dummies treats it as category
    df["Area code"] = df["Area code"].astype(str)

    # 2. One-hot encode categorical columns
    df_ohe = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # 3. Ensure every expected OHE column exists
    ohe_expected = [
        "International plan_Yes",
        "Voice mail plan_Yes",
        "Area code_415",
        "Area code_510",
    ]
    for col in ohe_expected:
        if col not in df_ohe.columns:
            df_ohe[col] = 0

    # Convert bool columns produced by get_dummies to int
    bool_cols = df_ohe.select_dtypes(include="bool").columns
    df_ohe[bool_cols] = df_ohe[bool_cols].astype(int)

    # 4. Scale numerical columns
    # Guard: only scale columns that exist in df_ohe
    scale_targets = [c for c in NUMERICAL_COLS if c in df_ohe.columns]
    df_ohe[scale_targets] = scaler.transform(df_ohe[scale_targets])

    # 5. Feature engineering on scaled data
    df_fe = create_powerful_features(df_ohe)

    # 6. Align columns to model expectations
    final = pd.DataFrame(0, index=df_fe.index, columns=feature_columns)
    for col in df_fe.columns:
        if col in final.columns:
            final[col] = df_fe[col].values
    final = final[feature_columns]

    return final


# ================================================================
# SIDEBAR — CUSTOMER INPUT
# ================================================================
st.sidebar.markdown("## 📋 Customer Profile")

def sidebar_section(title):
    st.sidebar.markdown(
        f"<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.06em;"
        f"text-transform:uppercase;color:#4a5568;margin:14px 0 4px;'>"
        f"{title}</p>", unsafe_allow_html=True
    )

sidebar_section("Account")
account_length        = st.sidebar.slider("Account Length (days)", 1, 250, 100)
area_code             = st.sidebar.selectbox("Area Code", [408, 415, 510])
international_plan    = st.sidebar.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan       = st.sidebar.selectbox("Voice Mail Plan",     ["No", "Yes"])
number_vmail_messages = st.sidebar.slider("Voicemail Messages",  0, 60, 10)
customer_service_calls= st.sidebar.slider("Customer Service Calls", 0, 10, 1)

sidebar_section("Daytime Usage")
total_day_minutes = st.sidebar.slider("Day Minutes",  0.0, 400.0, 180.0, 0.5)
total_day_calls   = st.sidebar.slider("Day Calls",    0,   200,   100)
total_day_charge  = st.sidebar.slider("Day Charge ($)", 0.0, 65.0, 30.6, 0.1)

sidebar_section("Evening Usage")
total_eve_minutes = st.sidebar.slider("Evening Minutes",   0.0, 400.0, 200.0, 0.5)
total_eve_calls   = st.sidebar.slider("Evening Calls",     0,   200,   100)
total_eve_charge  = st.sidebar.slider("Evening Charge ($)", 0.0, 35.0, 17.0, 0.1)

sidebar_section("Night Usage")
total_night_minutes = st.sidebar.slider("Night Minutes",    0.0, 400.0, 200.0, 0.5)
total_night_calls   = st.sidebar.slider("Night Calls",      0,   200,   100)
total_night_charge  = st.sidebar.slider("Night Charge ($)", 0.0, 25.0,  9.0, 0.1)

sidebar_section("International Usage")
total_intl_minutes = st.sidebar.slider("Intl Minutes",   0.0, 30.0,  10.0, 0.1)
total_intl_calls   = st.sidebar.slider("Intl Calls",     0,   20,    5)
total_intl_charge  = st.sidebar.slider("Intl Charge ($)",0.0, 10.0,  2.7, 0.1)


# ================================================================
# BUILD INPUT DATAFRAME
# ================================================================
input_data = {
    "Account length":          [account_length],
    "Area code":               [area_code],
    "International plan":      [international_plan],
    "Voice mail plan":         [voice_mail_plan],
    "Number vmail messages":   [number_vmail_messages],
    "Total day minutes":       [total_day_minutes],
    "Total day calls":         [total_day_calls],
    "Total day charge":        [total_day_charge],
    "Total eve minutes":       [total_eve_minutes],
    "Total eve calls":         [total_eve_calls],
    "Total eve charge":        [total_eve_charge],
    "Total night minutes":     [total_night_minutes],
    "Total night calls":       [total_night_calls],
    "Total night charge":      [total_night_charge],
    "Total intl minutes":      [total_intl_minutes],
    "Total intl calls":        [total_intl_calls],
    "Total intl charge":       [total_intl_charge],
    "Customer service calls":  [customer_service_calls],
}
input_df = pd.DataFrame(input_data)


# ================================================================
# MAIN PAGE
# ================================================================
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("📞 Telecom Customer Churn Predictor")
    st.caption("AI-powered churn risk assessment for telecom customer management")
with col_badge:
    if IS_DEMO:
        st.warning("⚡ Demo model active\n(place .joblib files to use your trained model)")
    else:
        st.success("✅ Production model loaded")

st.markdown("---")

# ── Input preview ────────────────────────────────────────────────
with st.expander("📋 Customer Input Summary", expanded=False):
    display_df = input_df.copy()
    st.dataframe(display_df.T.rename(columns={0: "Value"}),
                 use_container_width=True)

# ── Predict button ───────────────────────────────────────────────
predict_clicked = st.button(
    "🚀 Predict Churn Risk",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    try:
        X = preprocess(input_df)
        prediction   = model.predict(X)[0]
        probabilities= model.predict_proba(X)[0]
        churn_prob   = float(probabilities[1])
        retain_prob  = float(probabilities[0])

        # ── Risk tier ────────────────────────────────────────────
        if churn_prob >= 0.70:
            risk_level  = "HIGH"
            risk_color  = "#fc8181"
            risk_emoji  = "🔴"
        elif churn_prob >= 0.40:
            risk_level  = "MEDIUM"
            risk_color  = "#f6ad55"
            risk_emoji  = "🟡"
        else:
            risk_level  = "LOW"
            risk_color  = "#68d391"
            risk_emoji  = "🟢"

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        # ── Verdict banner ────────────────────────────────────────
        if prediction == 1:
            st.markdown(
                f"<div class='badge-churn'>⚠️ Customer is LIKELY TO CHURN — "
                f"Risk: {risk_emoji} {risk_level}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='badge-stay'>✅ Customer is LIKELY TO STAY — "
                f"Risk: {risk_emoji} {risk_level}</div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top KPI metrics ───────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Churn Probability",     f"{churn_prob*100:.2f}%")
        k2.metric("Retention Probability", f"{retain_prob*100:.2f}%")
        k3.metric("Risk Level",            risk_level)

        monthly_rev = (total_day_charge + total_eve_charge
                       + total_night_charge + total_intl_charge)
        k4.metric("Est. Monthly Revenue",  f"${monthly_rev:.2f}")

        st.markdown("---")

        # ── Charts row ────────────────────────────────────────────
        c_gauge, c_bar = st.columns([1, 1])

        # Gauge chart
        with c_gauge:
            st.markdown("**Churn Risk Gauge**")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_prob * 100,
                number={"suffix": "%", "font": {"size": 36}},
                delta={"reference": 50,
                       "increasing": {"color": "#fc8181"},
                       "decreasing": {"color": "#68d391"}},
                title={"text": "Churn Risk", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickcolor": "#4a5568"},
                    "bar":  {"color": risk_color, "thickness": 0.22},
                    "bgcolor": "#1a1f2e",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],  "color": "#1c2d22"},
                        {"range": [40, 70], "color": "#2d2618"},
                        {"range": [70, 100],"color": "#2d1a1a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": churn_prob * 100,
                    },
                },
            ))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                height=280,
                margin=dict(t=40, b=0, l=20, r=20),
            )
            st.plotly_chart(gauge, use_container_width=True)

        # Usage breakdown bar chart
        with c_bar:
            st.markdown("**Usage Breakdown**")
            usage_df = pd.DataFrame({
                "Period":  ["Day", "Evening", "Night", "International"],
                "Minutes": [total_day_minutes, total_eve_minutes,
                            total_night_minutes, total_intl_minutes],
                "Charge":  [total_day_charge, total_eve_charge,
                            total_night_charge, total_intl_charge],
            })
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Minutes",
                x=usage_df["Period"],
                y=usage_df["Minutes"],
                marker_color="#63b3ed",
                yaxis="y1",
            ))
            fig_bar.add_trace(go.Scatter(
                name="Charge ($)",
                x=usage_df["Period"],
                y=usage_df["Charge"],
                mode="lines+markers",
                marker_color="#f6ad55",
                marker_size=8,
                yaxis="y2",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                height=280,
                margin=dict(t=10, b=40, l=0, r=40),
                legend=dict(orientation="h", y=1.1),
                yaxis =dict(title="Minutes", gridcolor="#1e2535",
                            color="#e2e8f0"),
                yaxis2=dict(title="Charge ($)", overlaying="y",
                            side="right", color="#f6ad55"),
                xaxis =dict(gridcolor="#1e2535", color="#e2e8f0"),
                barmode="group",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Feature importance (proxy) radar ─────────────────────
        st.markdown("---")
        col_radar, col_rec = st.columns([1, 1])

        with col_radar:
            st.markdown("**Risk Factor Radar**")

            # Normalised [0,1] risk scores per dimension
            csc_score     = min(customer_service_calls / 10, 1.0)
            intl_score    = 1.0 if international_plan == "Yes" else 0.0
            day_min_score = min(total_day_minutes / 400, 1.0)
            vm_score      = 1.0 if voice_mail_plan == "No" else 0.0
            tenure_score  = max(0, 1 - account_length / 250)

            categories = ["CS Calls", "Intl Plan", "Day Usage",
                          "No Voicemail", "New Tenure"]
            scores     = [csc_score, intl_score, day_min_score,
                          vm_score, tenure_score]

            radar = go.Figure(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(252,129,129,0.15)",
                line=dict(color="#fc8181", width=2),
                marker=dict(size=6, color="#fc8181"),
            ))
            radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="#2d3748",
                                    color="#718096",
                                    tickvals=[0.25, 0.5, 0.75, 1.0],
                                    ticktext=["25%","50%","75%","100%"]),
                    angularaxis=dict(gridcolor="#2d3748", color="#a0aec0"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                height=300,
                margin=dict(t=30, b=10, l=50, r=50),
                showlegend=False,
            )
            st.plotly_chart(radar, use_container_width=True)

        with col_rec:
            st.markdown("**💡 Retention Recommendations**")

            if churn_prob >= 0.70:
                st.error("High churn risk detected")
                recs = [
                    ("🎁", "Offer personalised loyalty discount (10–20%)"),
                    ("📞", "Schedule priority retention call within 24 hours"),
                    ("👤", "Assign dedicated account manager"),
                    ("📦", "Provide free plan upgrade or bonus data"),
                    ("⚡", "Resolve any open support issues immediately"),
                ]
            elif churn_prob >= 0.40:
                st.warning("Moderate churn risk detected")
                recs = [
                    ("🏅", "Enrol in loyalty rewards programme"),
                    ("📋", "Review current plan vs better-fit alternatives"),
                    ("💬", "Send satisfaction survey + follow up personally"),
                    ("🎯", "Offer seasonal promotion or bundled service"),
                ]
            else:
                st.success("Low churn risk — customer appears stable")
                recs = [
                    ("💎", "Identify premium upsell opportunity"),
                    ("📣", "Invite to referral programme"),
                    ("📈", "Offer loyalty tier upgrade"),
                    ("✉️",  "Send periodic value-check communications"),
                ]

            for emoji, text in recs:
                st.markdown(f"- {emoji} {text}")

        # ── Detailed metrics table ────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Detailed Signal Analysis")

        total_mins   = (total_day_minutes + total_eve_minutes
                        + total_night_minutes + total_intl_minutes)
        total_calls  = (total_day_calls + total_eve_calls
                        + total_night_calls + total_intl_calls)
        avg_cpm      = monthly_rev / total_mins if total_mins > 0 else 0.0

        signals = pd.DataFrame({
            "Signal": [
                "Total Monthly Charge",
                "Total Minutes",
                "Total Calls",
                "Avg Charge / Minute",
                "Customer Service Calls",
                "International Plan",
                "Voice Mail Plan",
                "Account Length (days)",
            ],
            "Value": [
                f"${monthly_rev:.2f}",
                f"{total_mins:.0f} min",
                f"{total_calls}",
                f"${avg_cpm:.4f}",
                f"{customer_service_calls}",
                international_plan,
                voice_mail_plan,
                f"{account_length}",
            ],
            "Churn Signal": [
                "⚠️ High" if monthly_rev > 70 else "✅ Normal",
                "⚠️ High" if total_mins > 700 else "✅ Normal",
                "✅ Normal",
                "⚠️ High" if avg_cpm > 0.095 else "✅ Normal",
                ("🔴 Very High" if customer_service_calls >= 4
                 else "⚠️ Elevated" if customer_service_calls >= 2
                 else "✅ Normal"),
                "⚠️ Risk Factor" if international_plan == "Yes" else "✅ Normal",
                "✅ Protective"  if voice_mail_plan   == "Yes" else "⚠️ No plan",
                ("✅ Loyal" if account_length > 150
                 else "⚠️ Newer" if account_length < 60
                 else "✅ Normal"),
            ],
        })
        st.dataframe(signals, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"**Prediction failed:** {e}")
        with st.expander("🐛 Full traceback"):
            import traceback
            st.code(traceback.format_exc())

# ================================================================
# BATCH PREDICTION (CSV + LOCAL FILES SUPPORT)
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- LOCAL FILE PATHS ---------------- #
train_file = "churn-bigml-80.csv"
test_file  = "churn-bigml-20.csv"

# ================================================================
# BATCH PREDICTION SECTION
# ================================================================
st.markdown("---")
st.subheader("📂 Batch Prediction (CSV Upload + Local Files)")
st.caption("Upload your CSV OR use built-in dataset (80/20 split files)")

# ---------------- OPTIONS ---------------- #
mode = st.radio("Choose Input Mode:", ["Upload CSV", "Use Local Dataset"])

batch_df = None

# ---------------- UPLOAD OPTION ---------------- #
if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

# ---------------- LOCAL FILE OPTION ---------------- #
elif mode == "Use Local Dataset":
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Train Dataset (80%)", "Test Dataset (20%)"]
    )

    if dataset_choice == "Train Dataset (80%)":
        batch_df = pd.read_csv(train_file)
    else:
        batch_df = pd.read_csv(test_file)

# ================================================================
# RUN PREDICTIONS
# ================================================================
if batch_df is not None:

    try:
        st.write(f"Loaded **{len(batch_df):,}** rows × {len(batch_df.columns)} columns")

        with st.spinner("Running batch predictions…"):
            X_batch = preprocess(batch_df)   # must exist in your main code
            preds   = model.predict(X_batch)
            probs   = model.predict_proba(X_batch)[:, 1]

        # ---------------- RESULTS ---------------- #
        batch_df["Churn_Prediction"] = preds
        batch_df["Churn_Probability"] = np.round(probs * 100, 2)

        batch_df["Risk_Level"] = pd.cut(
            probs,
            bins=[0, 0.40, 0.70, 1.0],
            labels=["Low", "Medium", "High"]
        )

        # ---------------- METRICS ---------------- #
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", len(batch_df))
        c2.metric("Predicted Churners", int(preds.sum()))
        c3.metric("Churn Rate", f"{preds.mean()*100:.1f}%")

        # ---------------- TABLE ---------------- #
        st.dataframe(batch_df, use_container_width=True)

        # ---------------- CHART ---------------- #
        fig = px.histogram(
            batch_df,
            x="Churn_Probability",
            nbins=40,
            title="Churn Probability Distribution",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- DOWNLOAD ---------------- #
        csv_out = batch_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_out,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.caption("📞 Telecom Churn Predictor · Streamlit · ML Powered System")
