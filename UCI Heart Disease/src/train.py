import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

# ==========================
# CREATE FOLDERS
# ==========================

os.makedirs("artifacts", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data/heart.csv")

X = df.drop("condition", axis=1)
y = df["condition"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================
# FEATURE SCALING
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# SMOTE
# ==========================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nClass Distribution Before SMOTE")
print(y_train.value_counts())

print("\nClass Distribution After SMOTE")
print(y_train_smote.value_counts())

# ==========================
# MODEL COMPARISON
# ==========================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=42)
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\nMODEL COMPARISON\n")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_smote,
        y_train_smote,
        cv=cv,
        scoring="f1"
    )

    mean_score = scores.mean()

    results.append(
        [name, mean_score]
    )

    print(
        f"{name}: {mean_score:.4f}"
    )

comparison_df = pd.DataFrame(
    results,
    columns=["Model", "Mean_F1"]
)

comparison_df.to_csv(
    "artifacts/model_comparison.csv",
    index=False
)

# ==========================
# HYPERPARAMETER TUNING
# ==========================

print("\nTuning Random Forest...\n")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

rf = RandomForestClassifier(
    random_state=42
)

search = RandomizedSearchCV(
    rf,
    param_grid,
    n_iter=10,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

search.fit(
    X_train_smote,
    y_train_smote
)

best_model = search.best_estimator_

print("Best Parameters:")
print(search.best_params_)

# ==========================
# EVALUATION
# ==========================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_pred
)

print("\nFINAL RESULTS")
print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nClassification Report\n")
print(classification_report(
    y_test,
    y_pred
))

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig(
    "artifacts/confusion_matrix.png"
)

plt.close()

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(
    best_model,
    "models/heart_disease_model.joblib"
)

joblib.dump(
    scaler,
    "models/scaler.joblib"
)

print("\nModel Saved Successfully!")