import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/loan_default_prediction.csv")

# Features and target
X = df.drop(columns=["loan_id", "default"])
y = df["default"]

# Load preprocessor
preprocessor = joblib.load("models/preprocessor.joblib")

# Transform data
X = preprocessor.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
model = joblib.load("models/loan_default_model.joblib")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("Loan Default Prediction")
plt.savefig("artifacts/confusion_matrix.png")
plt.show()

print("\nConfusion Matrix saved to artifacts/confusion_matrix.png")