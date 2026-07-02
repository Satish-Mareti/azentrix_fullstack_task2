import joblib

print("Loading model...")
model = joblib.load("models/loan_default_model.joblib")
print("Model loaded successfully!")

print("Loading preprocessor...")
preprocessor = joblib.load("models/preprocessor.joblib")
print("Preprocessor loaded successfully!")