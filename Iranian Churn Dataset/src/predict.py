import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "artifacts" / "best_model.joblib"
)

scaler = joblib.load(
    BASE_DIR / "artifacts" / "scaler.joblib"
)

features = [
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
]

print("\n=== Customer Churn Prediction ===\n")

data = {}

for feature in features:
    value = float(input(f"{feature}: "))
    data[feature] = value

sample = pd.DataFrame([data])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]

if prediction == 1:
    print("\n⚠ Customer Likely To Churn")
else:
    print("\n✅ Customer Likely To Stay")