import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/loan_default_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    return model, preprocessor

model, preprocessor = load_artifacts()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🏦 Loan Default Prediction Dashboard")
st.markdown("""
Predict whether a customer is likely to default on a loan using a Machine Learning model trained on historical lending data.
            
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📋 Customer Information")

income = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=0,
    value=10000,
    step=500
)

loan_amount = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=0,
    value=20000,
    step=500
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["Employed", "Unemployed"]
)

# =========================
# DISPLAY INPUTS
# =========================
st.subheader("📊 Applicant Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Income", f"${income:,}")

with col2:
    st.metric("Loan Amount", f"${loan_amount:,}")

with col3:
    st.metric("Employment", employment_status)

# =========================
# RISK INDICATOR
# =========================
loan_ratio = loan_amount / max(income, 1)

if loan_ratio < 2:
    risk_text = "Low"
elif loan_ratio < 4:
    risk_text = "Medium"
else:
    risk_text = "High"

st.info(f"Estimated Loan-to-Income Risk Level: **{risk_text}**")

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Loan Default"):

    input_df = pd.DataFrame({
        "income": [income],
        "loan_amount": [loan_amount],
        "employment_status": [employment_status]
    })

    processed_data = preprocessor.transform(input_df)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0]

    default_prob = probability[1] * 100
    safe_prob = probability[0] * 100

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ HIGH RISK CUSTOMER\n\n"
            f"Probability of Default: {default_prob:.2f}%"
        )
    else:
        st.success(
            f"✅ LOW RISK CUSTOMER\n\n"
            f"Probability of Repayment: {safe_prob:.2f}%"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Default Probability",
            f"{default_prob:.2f}%"
        )

    with col2:
        st.metric(
            "Repayment Probability",
            f"{safe_prob:.2f}%"
        )

    st.progress(float(default_prob / 100))

# =========================
# ABOUT SECTION
# =========================
st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.write("""
    **Loan Default Prediction System**

    This project predicts whether a borrower is likely to default on a loan.

    ### Machine Learning Pipeline

    - Data Cleaning
    - Exploratory Data Analysis (EDA)
    - Feature Engineering
    - Class Balancing using SMOTE
    - Model Training & Evaluation
    - Cross Validation
    - Hyperparameter Optimization
    - Streamlit Deployment

    ### Models Compared

    - Logistic Regression
    - Random Forest
    - XGBoost

    ### Target Variable

    - 0 → No Default
    - 1 → Default
    """)


