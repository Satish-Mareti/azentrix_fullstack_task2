import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/loan_default_prediction.csv")

# Features and target
X = df.drop(columns=["loan_id", "default"])
y = df["default"]

# Column types
num_cols = ["income", "loan_amount"]
cat_cols = ["employment_status"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        )
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Transform
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Models
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

results = []

for name, model in models.items():

    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    ).mean()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\n" + "="*50)
    print(name)
    print("="*50)

    print(f"CV Score: {cv_score:.4f}")
    print(f"Accuracy: {acc:.4f}")

    print(classification_report(y_test, preds))
    results.append([name, cv_score, acc])

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

comparison_df = pd.DataFrame(
    results,
    columns=["Model", "CV Score", "Accuracy"]
)

comparison_df.to_csv(
    "artifacts/model_comparison.csv",
    index=False
)

print("\nModel Comparison:")
print(comparison_df)

# Save model
joblib.dump(best_model, "models/loan_default_model.joblib")
joblib.dump(preprocessor, "models/preprocessor.joblib")

print("\nBest Accuracy:", best_accuracy)
print("Model saved successfully.")