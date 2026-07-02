import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Iranian Churn Prediction",
    page_icon="📞",
    layout="wide"
)

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "best_model.joblib"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.joblib"

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📞 Telecom Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 Prediction",
        "ℹ About"
    ]
)
if page == "🏠 Home":

    st.markdown("""
    <div class='glass'>

    <h1>
    🚀 Iranian Telecom Churn Prediction
    </h1>

    <h4>
    End-to-End Machine Learning Pipeline
    </h4>

    <p>
    SMOTE • Cross Validation • Hyperparameter Tuning • XGBoost
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric("Customers","3150")

    with c2:
        st.metric("Features","13")

    with c3:
        st.metric("Accuracy","96.35%")

    with c4:
        st.metric("ROC-AUC","99.04%")

    st.markdown("---")

    st.subheader("🏆 Model Performance")

    comparison_df = pd.read_csv(
        BASE_DIR / "artifacts" / "model_comparison.csv"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.markdown("---")

    st.info("""

    """)

# ==========================================
# PREDICTION
# ==========================================

elif page == "🤖 Prediction":

    st.title("🤖 Customer Churn Prediction")

    call_failure = st.number_input("Call Failure", 0, 50, 5)
    complains = st.number_input("Complains", 0, 1, 0)
    subscription_length = st.number_input("Subscription Length", 1, 60, 20)
    charge_amount = st.number_input("Charge Amount", 0, 10, 1)
    seconds_of_use = st.number_input("Seconds of Use", 0, 20000, 1000)
    frequency_use = st.number_input("Frequency of use", 0, 1000, 100)
    frequency_sms = st.number_input("Frequency of SMS", 0, 1000, 20)
    distinct_called = st.number_input("Distinct Called Numbers", 0, 500, 20)
    age_group = st.number_input("Age Group", 1, 5, 2)
    tariff_plan = st.number_input("Tariff Plan", 1, 2, 1)
    status = st.number_input("Status", 1, 3, 1)
    age = st.number_input("Age", 15, 80, 30)
    customer_value = st.number_input("Customer Value", 0.0, 5000.0, 100.0)

    if st.button("Predict Churn"):

        data = pd.DataFrame([[
            call_failure,
            complains,
            subscription_length,
            charge_amount,
            seconds_of_use,
            frequency_use,
            frequency_sms,
            distinct_called,
            age_group,
            tariff_plan,
            status,
            age,
            customer_value
        ]], columns=[
            "Call Failure",
            "Complains",
            "Subscription Length",
            "Charge Amount",
            "Seconds of Use",
            "Frequency of use",
            "Frequency of SMS",
            "Distinct Called Numbers",
            "Age Group",
            "Tariff Plan",
            "Status",
            "Age",
            "Customer Value"
        ])

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        st.progress(float(probability))

        if prediction == 1:
            st.error(
                f"⚠ Customer Likely To Churn\n\nConfidence: {probability:.2%}"
            )
        else:
            st.success(
                f"✅ Customer Likely To Stay\n\nConfidence: {(1-probability):.2%}"
            )

# ==========================================
# ABOUT
# ==========================================

elif page == "ℹ About":

    st.title("About Project")

    st.write("""
    Customer Churn Prediction Using Machine Learning

    Dataset:
    Iranian Churn Dataset (UCI)

    Models:
    - Logistic Regression
    - Random Forest
    - XGBoost

    Techniques:
    - SMOTE
    - Hyperparameter Tuning
    - Cross Validation

    Developer:
    Mareti Satish
    """)