import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/loan_default_prediction.csv")

X = df.drop(columns=["loan_id", "default"])
y = df["default"]

preprocessor = joblib.load(
    "models/preprocessor.joblib"
)

X = preprocessor.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"]
}

search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000),
    params,
    cv=5,
    scoring="accuracy",
    n_iter=10,
    random_state=42
)

search.fit(X_train, y_train)

print("Best Parameters:")
print(search.best_params_)

print("\nBest Score:")
print(search.best_score_)