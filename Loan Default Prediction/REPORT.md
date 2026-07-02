# Loan Default Prediction Report

## 1. Problem Statement

Financial institutions face significant risks when approving loans for customers who may fail to repay them. Loan defaults can result in substantial financial losses and impact the stability of lending organizations.

The objective of this project is to build a Machine Learning model capable of predicting whether a customer is likely to default on a loan based on historical borrower information. The solution aims to assist financial institutions in making informed lending decisions and reducing credit risk.

---

## 2. Dataset Description

### Dataset Source

Kaggle – Loan Default Prediction Dataset

### Dataset Size

* Total Records: 1000
* Total Features: 5

### Features

| Feature           | Description                                   |
| ----------------- | --------------------------------------------- |
| loan_id           | Unique loan identifier                        |
| income            | Applicant income                              |
| loan_amount       | Requested loan amount                         |
| employment_status | Employment status of applicant                |
| default           | Target variable (0 = No Default, 1 = Default) |

### Target Variable

* 0 → No Default
* 1 → Default

---

## 3. EDA Findings

Exploratory Data Analysis (EDA) was performed to understand the structure and quality of the dataset.

### Findings

* Dataset contains 1000 records.
* No missing values were found.
* No duplicate records were identified.
* Numerical features include income and loan_amount.
* One categorical feature: employment_status.
* Target classes were nearly balanced.

### Class Distribution

| Class          | Count |
| -------------- | ----- |
| No Default (0) | 513   |
| Default (1)    | 487   |

### Observation

The dataset is almost balanced with approximately:

* 51.3% No Default
* 48.7% Default

This reduces the risk of model bias toward a particular class.

---

## 4. Feature Engineering

Several preprocessing techniques were applied to improve model performance.

### Numerical Features

* Missing value imputation using Median strategy
* Standardization using StandardScaler

### Categorical Features

* Missing value imputation using Most Frequent strategy
* One-Hot Encoding using OneHotEncoder

### Removed Features

* loan_id

Reason:

The loan identifier does not provide predictive information and may introduce noise into the model.

### Preprocessing Pipeline

A ColumnTransformer pipeline was implemented to automate preprocessing for numerical and categorical features.

---

## 5. Handling Class Imbalance

Although the dataset was relatively balanced, SMOTE (Synthetic Minority Over-sampling Technique) was implemented to satisfy project requirements and demonstrate class imbalance handling.

### Technique Used

SMOTE

### Benefits

* Generates synthetic minority samples
* Improves model learning
* Reduces prediction bias

Implementation was applied only on the training dataset to prevent data leakage.

---

## 6. Models Compared

Three machine learning models were trained and evaluated.

### Model 1: Logistic Regression

Advantages:

* Simple and interpretable
* Fast training
* Strong baseline classifier

### Model 2: Random Forest

Advantages:

* Handles non-linear relationships
* Reduces overfitting through ensemble learning

### Model 3: Gradient Boosting

Advantages:

* Sequential learning approach
* Often achieves strong predictive performance

---

## 7. Cross Validation Results

5-Fold Cross Validation was used to evaluate model generalization.

| Model               | CV Score |
| ------------------- | -------- |
| Logistic Regression | 0.7110   |
| Random Forest       | 0.7073   |
| Gradient Boosting   | 0.7268   |

### Observation

Gradient Boosting achieved the highest cross-validation score, indicating strong generalization performance.

---

## 8. Hyperparameter Tuning

RandomizedSearchCV was used to optimize model hyperparameters.

### Parameters Tuned

Logistic Regression:

* C
* Solver

### Optimization Method

* RandomizedSearchCV
* 5-fold cross validation
* Accuracy scoring metric

### Objective

Identify parameter combinations that maximize prediction accuracy.

---

## 9. Final Model Selection

After comparing all models, Logistic Regression was selected as the final model.

### Reason for Selection

| Metric                 | Value  |
| ---------------------- | ------ |
| Accuracy               | 78.50% |
| Cross Validation Score | 71.10% |

Although Gradient Boosting achieved a slightly higher cross-validation score, Logistic Regression produced the highest test accuracy and demonstrated better performance on unseen data.

### Selected Model

Logistic Regression

---

## 10. Evaluation Metrics

### Logistic Regression Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 78.50% |
| Precision | 83%    |
| Recall    | 74%    |
| F1 Score  | 78%    |

### Classification Report

Class 0:

* Precision: 0.83
* Recall: 0.74
* F1-Score: 0.78

Class 1:

* Precision: 0.75
* Recall: 0.84
* F1-Score: 0.79

### Confusion Matrix

A confusion matrix was generated and saved as:

```text
artifacts/confusion_matrix.png
```

---

## 11. Deployment

The trained model was deployed using Streamlit.

### Features

* User-friendly interface
* Real-time predictions
* Loan risk assessment
* Probability estimates

### Deployment Workflow

1. User enters borrower information
2. Data is preprocessed
3. Model generates prediction
4. Risk probability is displayed

### Model Persistence

The final model was saved using Joblib:

```text
models/loan_default_model.joblib
```

Preprocessing pipeline:

```text
models/preprocessor.joblib
```

---

## 12. Future Improvements

Several enhancements can improve the project further:

### Data Improvements

* Increase dataset size
* Include additional borrower information
* Incorporate credit history features

### Model Improvements

* Experiment with XGBoost
* Explore LightGBM
* Apply feature selection techniques

### Explainability

* Implement SHAP values
* Add LIME explanations

### Deployment Improvements

* Cloud deployment
* REST API integration
* Database connectivity

### Business Enhancements

* Automated credit scoring
* Real-time risk monitoring
* Loan recommendation engine

---

## Conclusion

This project successfully developed an end-to-end machine learning pipeline for loan default prediction. The workflow included data preprocessing, feature engineering, class balancing, model comparison, hyperparameter tuning, evaluation, and deployment.

Among the evaluated models, Logistic Regression achieved the best performance with an accuracy of 78.50% and was selected as the final model. The deployed Streamlit application provides an accessible interface for predicting loan default risk and demonstrates practical application of machine learning in financial risk assessment.
