"""
Streamlit Artifact: Customer Churn Predictor
----------------------------------------------
Loads a previously trained & saved model (Logistic Regression or CatBoost),
collects customer feature inputs through a form, and displays the churn
prediction, probability, and a risk gauge.

Run with:
    streamlit run churn_predictor_app.py

Expected model files (produced earlier by save_model()):
    saved_models/logistic_regression.joblib
    saved_models/logistic_regression_preprocessor.joblib
    saved_models/catboost_vanilla.cbm
    saved_models/catboost_optimized.cbm
"""

import os
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
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------
# Path resolution: anchor to this script's location, not the process's
# working directory. This matters for Streamlit Community Cloud, which
# clones the GitHub repo and may invoke the app from a different cwd
# depending on repo layout / entry-point settings.
# ----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(APP_DIR, "saved_models")

# Fail fast and clearly if the models directory wasn't committed to the repo
# (the most common deployment mistake: .gitignore excluding *.joblib/*.cbm,
# or forgetting `git add saved_models/`).
if not os.path.isdir(SAVE_DIR):
    st.error(
        f"Could not find the 'saved_models' folder at: {SAVE_DIR}\n\n"
        f"This usually means the model files weren't committed to the GitHub "
        f"repo, or a .gitignore rule is excluding them. Run `git status` "
        f"locally to confirm saved_models/*.joblib and *.cbm are tracked, "
        f"then `git add saved_models/ && git commit && git push`."
    )
    st.stop()

# Registry of selectable models: maps display name -> (file path, model kind)
MODEL_REGISTRY = {
    "Logistic Regression": {
        "path": os.path.join(SAVE_DIR, "logistic_regression.joblib"),
        "kind": "sklearn",
        "preprocessor_path": os.path.join(SAVE_DIR, "logistic_regression_preprocessor.joblib"),
    },
    "CatBoost (Vanilla)": {
        "path": os.path.join(SAVE_DIR, "catboost_vanilla.cbm"),
        "kind": "catboost",
        "preprocessor_path": None,
    },
    "CatBoost (Optimized)": {
        "path": os.path.join(SAVE_DIR, "catboost_optimized.cbm"),
        "kind": "catboost",
        "preprocessor_path": None,
    },
}

# Categorical options pulled from the IBM Telco Churn schema
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

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Column order must match the X dataframe used during training
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_selected_model(model_key):
    """Load model (and preprocessor, if any) for the chosen registry entry."""
    entry = MODEL_REGISTRY[model_key]
    model_path = entry["path"]

    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"

    if entry["kind"] == "sklearn":
        model = joblib.load(model_path)
        preprocessor = None
        if entry["preprocessor_path"]:
            if not os.path.exists(entry["preprocessor_path"]):
                return None, None, f"Preprocessor file not found: {entry['preprocessor_path']}"
            preprocessor = joblib.load(entry["preprocessor_path"])
        return model, preprocessor, None

    elif entry["kind"] == "catboost":
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model, None, None

    return None, None, "Unknown model kind."


# ----------------------------------------------------------------------
# Sidebar: model selection
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Model Settings")
selected_model_key = st.sidebar.selectbox(
    "Choose a trained model",
    options=list(MODEL_REGISTRY.keys()),
)

model, preprocessor, load_error = load_selected_model(selected_model_key)

if load_error:
    st.sidebar.error(load_error)
else:
    st.sidebar.success(f"Loaded: {selected_model_key}")

with st.sidebar.expander("🔍 Debug: loaded object types"):
    st.write("model path:", MODEL_REGISTRY[selected_model_key]["path"])
    st.write("model type:", type(model))
    st.write("preprocessor path:", MODEL_REGISTRY[selected_model_key]["preprocessor_path"])
    st.write("preprocessor type:", type(preprocessor))
    if st.button("Clear cache & reload models"):
        st.cache_resource.clear()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models are loaded from the local `saved_models/` directory. "
    "Train and save models first using `save_model()` from your training script."
)

# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
st.title("📊 Customer Churn Predictor")
st.markdown(
    "Enter customer details below to estimate the probability of churn "
    "using the selected model."
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
        st.error("No model is loaded. Check the sidebar for the file error.")
    else:
        # Assemble a single-row dataframe matching the training feature order
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
            if preprocessor is not None:
                if not hasattr(preprocessor, "transform"):
                    st.error(
                        f"Loaded 'preprocessor' is actually a {type(preprocessor).__name__}, "
                        f"not a fitted transformer. The saved file at "
                        f"'{MODEL_REGISTRY[selected_model_key]['preprocessor_path']}' likely "
                        f"contains the wrong object — re-run the save_model() step for this "
                        f"model in your notebook, then click 'Clear cache & reload models' "
                        f"in the sidebar."
                    )
                    st.stop()
                # Logistic Regression path: transform raw input, then predict
                X_input = preprocessor.transform(input_row)
                prob_churn = model.predict_proba(X_input)[:, 1][0]
            else:
                # CatBoost path: model consumes raw categorical strings directly
                prob_churn = model.predict_proba(input_row)[:, 1][0]

            pred_label = "Will Churn" if prob_churn >= 0.5 else "Will Stay"
            pred_color = "#d63031" if prob_churn >= 0.5 else "#00b894"

            st.markdown(
                f"<h2 style='color:{pred_color};'>{pred_label}</h2>",
                unsafe_allow_html=True,
            )
            st.metric("Churn Probability", f"{prob_churn * 100:.1f}%")

            # Risk gauge
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
    "Model: Telco Customer Churn classifier · "
    "Input features mirror the IBM Telco Customer Churn dataset schema."
)
