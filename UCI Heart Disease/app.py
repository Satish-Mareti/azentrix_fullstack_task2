import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.main {
    background-color: #fff5f5;
}

.stButton>button {
    background-color: #dc2626;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #b91c1c;
}

.metric-card {
    background-color: #fee2e2;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.big-font {
    font-size:40px !important;
    color:#b91c1c;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------

model = joblib.load("models/heart_disease_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2382/2382461.png",
    width=120
)

st.sidebar.title("❤️ Heart Disease ML Pipeline")

st.sidebar.markdown("""
### Model Performance

Accuracy: 86.67%

F1 Score: 84.62%

ROC-AUC: 86.16%
""")

# -------------------------
# HEADER
# -------------------------

st.markdown(
"""
<h1 style='text-align:center;color:#b91c1c'>
❤️ Heart Disease Prediction Dashboard
</h1>
<p style='text-align:center;font-size:18px'>
Machine Learning Based Cardiovascular Risk Assessment
</p>
""",
unsafe_allow_html=True
)

st.divider()

# -------------------------
# INPUT FORM
# -------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0,1])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Blood Pressure", 80,250,120)
    chol = st.number_input("Cholesterol",100,600,220)
    fbs = st.selectbox("Fasting Blood Sugar", [0,1])

with col2:
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate",60,220,150)
    exang = st.selectbox("Exercise Angina",[0,1])
    oldpeak = st.number_input("Old Peak",0.0,10.0,1.0)
    slope = st.selectbox("Slope",[0,1,2])
    ca = st.selectbox("CA",[0,1,2,3,4])
    thal = st.selectbox("Thal",[0,1,2,3])

# -------------------------
# PREDICTION
# -------------------------

if st.button("🔍 Predict Heart Disease Risk"):

    input_df = pd.DataFrame([[
        age, sex, cp, trestbps,
        chol, fbs, restecg,
        thalach, exang, oldpeak,
        slope, ca, thal
    ]], columns=[
        'age','sex','cp','trestbps','chol',
        'fbs','restecg','thalach',
        'exang','oldpeak','slope',
        'ca','thal'
    ])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.divider()

    if prediction == 1:

        st.error(
            "⚠️ High Risk of Heart Disease Detected"
        )

    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )

    # -------------------------
    # MODEL METRICS
    # -------------------------

    st.subheader("📊 Model Performance")

    c1,c2,c3 = st.columns(3)

    c1.metric("Accuracy","86.67%")
    c2.metric("F1 Score","84.62%")
    c3.metric("ROC-AUC","86.16%")

    st.divider()

    # -------------------------
    # EDA IMAGES
    # -------------------------

    st.subheader("📈 Dataset Analysis")

    img1, img2 = st.columns(2)

    with img1:
        st.image(
            "artifacts/target_distribution.png",
            caption="Target Distribution"
        )

        st.image(
            "artifacts/age_vs_condition.png",
            caption="Age vs Heart Disease"
        )

    with img2:
        st.image(
            "artifacts/correlation_heatmap.png",
            caption="Correlation Heatmap"
        )

        st.image(
            "artifacts/chol_distribution.png",
            caption="Cholesterol Distribution"
        )

    st.divider()

    st.subheader("🧠 Confusion Matrix")

    st.image(
        "artifacts/confusion_matrix.png",
        use_container_width=True
    )

    st.divider()

    st.subheader("🏆 Model Comparison")

    comparison = pd.read_csv(
        "artifacts/model_comparison.csv"
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

st.divider()
