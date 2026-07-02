# 🏦 Loan Default Prediction

## 📌 Project Overview

This project builds an End-to-End Machine Learning Pipeline to predict whether a customer is likely to default on a loan. The goal is to assist financial institutions in identifying high-risk borrowers and making data-driven lending decisions.

The project covers the complete ML workflow:

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Class Imbalance Handling using SMOTE
* Model Training & Evaluation
* Cross Validation
* Hyperparameter Tuning
* Model Deployment using Streamlit

---

## 📊 Dataset Information

**Dataset:** Loan Default Prediction Dataset

**Source:** Kaggle

**Rows:** 1000

**Features:**

| Feature           | Description                                   |
| ----------------- | --------------------------------------------- |
| loan_id           | Unique Loan Identifier                        |
| income            | Customer Income                               |
| loan_amount       | Requested Loan Amount                         |
| employment_status | Employment Status                             |
| default           | Target Variable (0 = No Default, 1 = Default) |

---

## 🚀 Features

* Data Cleaning and Validation
* Exploratory Data Analysis (EDA)
* Feature Engineering
* One-Hot Encoding for Categorical Variables
* Standard Scaling for Numerical Features
* Class Balancing using SMOTE
* Comparison of Multiple Machine Learning Models
* Cross Validation
* Hyperparameter Tuning
* Model Persistence using Joblib
* Interactive Streamlit Dashboard

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
Loan Default Prediction/
│
├── artifacts/
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── dashboard.png
│
├── data/
│   └── loan_default_prediction.csv
│
├── models/
│   ├── loan_default_model.joblib
│   └── preprocessor.joblib
│
├── app.py
├── train.py
├── train_models.py
├── evaluate.py
├── hyperparameter_tuning.py
├── README.md
├── REPORT.md
├── requirements.txt
└── .gitignore
```

---

## 🤖 Model Comparison

| Model               | Cross Validation Score | Accuracy   |
| ------------------- | ---------------------- | ---------- |
| Logistic Regression | 0.7110                 | **78.50%** |
| Random Forest       | 0.7073                 | 69.00%     |
| Gradient Boosting   | 0.7268                 | 72.50%     |

### Best Model

**Logistic Regression**

Accuracy: **78.50%**

---

## 📈 Results

### Evaluation Metrics

* Accuracy: 78.50%
* Precision: 83%
* Recall: 74%
* F1 Score: 78%

### Generated Artifacts

* Confusion Matrix
* Model Comparison Report
* Saved Trained Model
* Streamlit Dashboard

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd loan-default-prediction
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train Models

```bash
py train_models.py
```

### Evaluate Model

```bash
py evaluate.py
```

### Hyperparameter Tuning

```bash
py hyperparameter_tuning.py
```

### Launch Streamlit Dashboard

```bash
py -m streamlit run app.py
```

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard where users can:

* Enter customer details
* Predict loan default risk
* View prediction probabilities
* Analyze borrower risk

### Dashboard Preview

Add your dashboard screenshot here:

```markdown
![Dashboard](artifacts/dashboard.png)
```

---

## 🔮 Future Improvements

* Collect larger and more diverse datasets
* Add additional borrower features
* Implement Explainable AI (SHAP/LIME)
* Deploy application on Streamlit Cloud
* Integrate real-time prediction API
* Experiment with Deep Learning Models

---

## 👨‍💻 Author

**Mareti Satish**

B.Tech CSE (AI & Computational Intelligence)

KL University
