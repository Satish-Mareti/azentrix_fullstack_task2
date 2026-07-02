import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Customer Churn.csv"
ARTIFACTS = BASE_DIR / "artifacts"
IMAGES = BASE_DIR / "images"

ARTIFACTS.mkdir(exist_ok=True)
IMAGES.mkdir(exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading Dataset...")

df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace("  ", " ")

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ==========================================
# EDA
# ==========================================

plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Class Distribution")
plt.savefig(IMAGES / "class_distribution.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(IMAGES / "correlation_heatmap.png")
plt.close()

# ==========================================
# FEATURES / TARGET
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    ARTIFACTS / "scaler.joblib"
)

# ==========================================
# SMOTE
# ==========================================

print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# ==========================================
# MODELS
# ==========================================

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
}

# ==========================================
# CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\nMODEL COMPARISON")
print("=" * 50)

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

    print(f"{name}: {mean_score:.4f}")

comparison_df = pd.DataFrame(
    results,
    columns=["Model", "CV_F1"]
)

comparison_df.to_csv(
    ARTIFACTS / "model_comparison.csv",
    index=False
)

# ==========================================
# HYPERPARAMETER TUNING
# ==========================================

print("\nRunning Hyperparameter Tuning...")
print("=" * 50)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0]
}

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
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

print("\nBest Parameters:")
print(search.best_params_)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = best_model.predict(X_test_scaled)

y_prob = best_model.predict_proba(
    X_test_scaled
)[:, 1]

# ==========================================
# METRICS
# ==========================================

print("\nFINAL RESULTS")
print("=" * 50)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

# ==========================================
# CONFUSION MATRIX
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("Confusion Matrix")
plt.savefig(
    IMAGES / "confusion_matrix.png"
)
plt.close()

# ==========================================
# ROC CURVE
# ==========================================

RocCurveDisplay.from_estimator(
    best_model,
    X_test_scaled,
    y_test
)

plt.title("ROC Curve")
plt.savefig(
    IMAGES / "roc_curve.png"
)
plt.close()

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    best_model,
    ARTIFACTS / "best_model.joblib"
)

joblib.dump(
    X.columns.tolist(),
    ARTIFACTS / "feature_columns.joblib"
)

print("\nModel Saved Successfully!")