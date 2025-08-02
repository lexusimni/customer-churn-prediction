import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------- Paths to artifacts saved by your notebook -------
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "model_xgb.pkl")
FEATS_PATH   = os.path.join(ARTIFACT_DIR, "feature_list.json")

# ------- Cache model + features to avoid reloading -------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATS_PATH)):
        raise FileNotFoundError(
            "Artifacts not found. Run the notebook to create "
            "`artifacts/model_xgb.pkl` and `artifacts/feature_list.json`."
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATS_PATH, "r") as f:
        features = json.load(f)
    return model, features

def align_columns(raw_df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Apply the same one-hot as training (drop_first=True) and align to
    the exact feature order used by the model. Any missing columns are added
    as zeros; any extra columns are dropped.
    """
    # Drop ID/target if present
    base = raw_df.drop(columns=[c for c in ["customerID", "Churn"] if c in raw_df.columns], errors="ignore")
    X = pd.get_dummies(base, drop_first=True)
    # Add missing feature columns as 0
    for col in features:
        if col not in X.columns:
            X[col] = 0
    # Keep only expected features and in the correct order
    X = X[features]
    return X

# ========================= UI =========================
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Predictor (XGBoost + SHAP-ready)")
st.write(
    "Upload a Telco-like CSV (raw columns) or try the quick example below. "
    "This app uses the trained XGBoost model from the notebook."
)

# Load artifacts
try:
    model, features = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

# ============== File Upload & Batch Prediction ==============
st.header("1) Upload CSV for Batch Predictions")
uploaded = st.file_uploader("Upload a CSV with columns like the original Telco dataset", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    st.write("**Preview of uploaded data:**")
    st.dataframe(raw.head())

    X = align_columns(raw, features)
    proba = model.predict_proba(X)[:, 1]

    out = raw.copy()
    out["churn_probability"] = proba
    st.success("Predictions generated!")
    st.write(out.head())

    st.download_button(
        "⬇️ Download predictions as CSV",
        data=out.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

# ============== Single Example Form ==============
st.header("2) Quick Single Prediction")
st.caption("Enter a few fields (others will be treated as defaults).")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=1.0)
    partner = st.selectbox("Partner", ["Yes", "No"])
with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

example_raw = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "Partner": partner,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "InternetService": internet
}])

X_ex = align_columns(example_raw, features)
p = float(model.predict_proba(X_ex)[:, 1][0])
st.metric("Predicted Churn Probability", f"{p:.2%}")

# ============== Optional: Local SHAP Explanation ==============
with st.expander("🔎 Show local explanation (SHAP) for this example"):
    try:
        import shap
        import matplotlib.pyplot as plt

        # Use TreeExplainer on the aligned row
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_ex)

        # Build a quick bar chart of top contributions
        vals = sv[0]
        df_imp = pd.DataFrame({"feature": X_ex.columns, "shap": vals}).copy()
        df_imp["abs"] = df_imp["shap"].abs()
        df_imp = df_imp.sort_values("abs", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(df_imp["feature"][::-1], df_imp["shap"][::-1])
        ax.set_title("Top SHAP contributions for this example")
        ax.set_xlabel("SHAP value (impact on churn log-odds)")
        st.pyplot(fig, clear_figure=True)

        st.caption("Note: Positive SHAP increases churn risk; negative decreases it.")
    except Exception as e:
        st.info("Install `shap` in the same environment to see local explanations.")
        st.code(f"{e}")
