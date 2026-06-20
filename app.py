"""
Streamlit Churn Predictor — CatBoost models loaded directly from GitHub
--------------------------------------------------------------------------
Fetches trained CatBoost model artifacts at runtime from:
    https://github.com/cchuang2009/ML-2026/tree/main/saved_models
via raw.githubusercontent.com, rather than relying on a local file clone.

Logistic Regression is intentionally excluded: it requires a fitted
preprocessor (ColumnTransformer) to encode raw categorical input, and no
such preprocessor file currently exists in the repo. CatBoost models
consume raw categorical strings natively, so no preprocessor is needed.

Run with:
    streamlit run churn_predictor_app.py
"""

import io
import os
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostClassifier

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="\U0001F4CA",
    layout="wide",
)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/cchuang2009/ML-2026/main/saved_models"

MODEL_REGISTRY = {
    "CatBoost (Vanilla)": {
        "model_file": "catboost_vanilla.cbm",
    },
    "CatBoost (Optimized)": {
        "model_file": "catboost_optimized.cbm",
    },
}

CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ----------------------------------------------------------------------
# GitHub model loaders (cached so each file is fetched only once per session)
# ----------------------------------------------------------------------
def fetch_bytes_from_github(filename, base_url=GITHUB_RAW_BASE, timeout=15):
    url = f"{base_url}/{filename}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content

    # Detect a broken git symlink masquerading as the model file: it downloads
    # as a short ASCII string (e.g. "../lr_model.joblib") instead of binary data.
    if len(content) < 200 and content.isascii():
        raise ValueError(
            f"'{filename}' downloaded as a tiny text file ({len(content)} bytes): "
            f"{content!r}. This looks like a broken git symlink rather than the "
            f"real model file. Fix it in the repo by re-adding the actual binary "
            f"(git rm + git add the real file, not a symlink), then commit & push."
        )
    return content


@st.cache_resource(show_spinner="Downloading model from GitHub...")
def load_selected_model(model_key):
    entry = MODEL_REGISTRY[model_key]

    try:
        model_bytes = fetch_bytes_from_github(entry["model_file"])
        tmp_path = os.path.join("/tmp", entry["model_file"])
        with open(tmp_path, "wb") as f:
            f.write(model_bytes)

        # Files in this repo may be saved as either CatBoost's native
        # binary format, or joblib/pickle despite the .cbm extension.
        # Try native first, fall back to joblib automatically.
        try:
            model = CatBoostClassifier()
            model.load_model(tmp_path)
            return model, None
        except Exception:
            model = joblib.load(io.BytesIO(model_bytes))
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"Loaded object is a {type(model).__name__}, not a usable model.")
            return model, None

    except requests.exceptions.RequestException as e:
        return None, f"Network error fetching '{entry['model_file']}' from GitHub: {e}"
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error loading '{entry['model_file']}': {e}"


# ----------------------------------------------------------------------
# Sidebar: model selection
# ----------------------------------------------------------------------
st.sidebar.title("\u2699\ufe0f Model Settings")
st.sidebar.caption(f"Models are fetched live from:\n`{GITHUB_RAW_BASE}`")

selected_model_key = st.sidebar.selectbox(
    "Choose a trained model",
    options=list(MODEL_REGISTRY.keys()),
)

model, load_error = load_selected_model(selected_model_key)

if load_error:
    st.sidebar.error(load_error)
else:
    st.sidebar.success(f"Loaded: {selected_model_key}")

if st.sidebar.button("\U0001F504 Reload model"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")

# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
st.title("\U0001F4CA Customer Churn Predictor")
st.markdown(
    "Enter customer details below to estimate the probability of churn. "
    "Models are loaded directly from the GitHub repository at runtime."
)

col_form, col_result = st.columns([1.2, 1], gap="large")

with col_form:
    st.subheader("Customer Profile")

    with st.form("input_form"):
        c1, c2 = st.columns(2)

        with c1:
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
            senior = st.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["SeniorCitizen"])
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
            dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"])
            phone_service = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"])
            multiple_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])
            internet_service = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
            online_security = st.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])
            online_backup = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])

        with c2:
            device_protection = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])
            tech_support = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
            streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
            streaming_movies = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless_billing = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"])
            payment_method = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)

        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=float(tenure) * monthly_charges, step=1.0
        )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

with col_result:
    st.subheader("Prediction Result")

    if not submitted:
        st.info("Fill in the customer profile and click **Predict Churn** to see results.")
    elif model is None:
        st.error("No model is loaded. Check the sidebar for the load error.")
    else:
        input_row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])[FEATURE_ORDER]

        try:
            prob_churn = model.predict_proba(input_row)[:, 1][0]

            pred_label = "Will Churn" if prob_churn >= 0.5 else "Will Stay"
            pred_color = "#d63031" if prob_churn >= 0.5 else "#00b894"

            st.markdown(
                f"<h2 style='color:{pred_color};'>{pred_label}</h2>",
                unsafe_allow_html=True,
            )
            st.metric("Churn Probability", f"{prob_churn * 100:.1f}%")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_churn * 100,
                number={"suffix": "%"},
                title={"text": "Churn Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": pred_color},
                    "steps": [
                        {"range": [0, 33], "color": "#dfe6e9"},
                        {"range": [33, 66], "color": "#ffeaa7"},
                        {"range": [66, 100], "color": "#fab1a0"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption(
    "Model: Telco Customer Churn classifier \u00b7 "
    "Models fetched live from github.com/cchuang2009/ML-2026/saved_models \u00b7 "
    "Input features mirror the IBM Telco Customer Churn dataset schema."
)
