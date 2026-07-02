import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/heart_disease_model.joblib")
scaler = joblib.load("models/scaler.joblib")

print("\n=== Heart Disease Prediction ===\n")

age = int(input("Age: "))
sex = int(input("Sex (0=Female, 1=Male): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (0/1): "))
restecg = int(input("Rest ECG (0-2): "))
thalach = int(input("Max Heart Rate: "))
exang = int(input("Exercise Angina (0/1): "))
oldpeak = float(input("Oldpeak: "))
slope = int(input("Slope (0-2): "))
ca = int(input("CA (0-4): "))
thal = int(input("Thal (0-3): "))

data = pd.DataFrame([[
    age, sex, cp, trestbps, chol,
    fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal
]], columns=[
    'age','sex','cp','trestbps','chol',
    'fbs','restecg','thalach','exang',
    'oldpeak','slope','ca','thal'
])

data_scaled = scaler.transform(data)

prediction = model.predict(data_scaled)[0]

print("\nPrediction Result")

if prediction == 1:
    print("⚠️ Heart Disease Detected")
else:
    print("✅ No Heart Disease")