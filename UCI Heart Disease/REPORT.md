# Heart Disease Prediction System - Project Report

## 1. Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection of cardiovascular disease can significantly improve treatment outcomes and reduce mortality rates.

The objective of this project is to develop an end-to-end Machine Learning Pipeline capable of predicting the presence of heart disease based on patient medical attributes. The project covers the complete machine learning workflow, including data analysis, feature engineering, class imbalance handling, model training, evaluation, hyperparameter tuning, and deployment through a Streamlit web application.

---

# 2. Dataset Description

Dataset: UCI Cleveland Heart Disease Dataset

Source:
https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

### Dataset Characteristics

* Number of Records: 297
* Number of Features: 13
* Target Variable: condition

Target Values:

* 0 → No Heart Disease
* 1 → Heart Disease Present

### Features

| Feature  | Description                       |
| -------- | --------------------------------- |
| age      | Age of patient                    |
| sex      | Gender                            |
| cp       | Chest pain type                   |
| trestbps | Resting blood pressure            |
| chol     | Cholesterol level                 |
| fbs      | Fasting blood sugar               |
| restecg  | Resting ECG results               |
| thalach  | Maximum heart rate achieved       |
| exang    | Exercise induced angina           |
| oldpeak  | ST depression induced by exercise |
| slope    | Slope of peak exercise ST segment |
| ca       | Number of major vessels           |
| thal     | Thalassemia                       |

---

# 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand feature distributions and identify relationships between variables.

### Analyses Performed

#### Target Distribution

The dataset contains both positive and negative heart disease cases. The classes are relatively balanced.

#### Correlation Analysis

A correlation heatmap was generated to identify relationships between medical attributes and the target variable.

#### Age vs Condition

Box plots were used to analyze the relationship between patient age and heart disease occurrence.

#### Cholesterol Distribution

The cholesterol values were visualized to understand their distribution across patients.

### EDA Outputs

* target_distribution.png
* correlation_heatmap.png
* age_vs_condition.png
* chol_distribution.png

---

# 4. Data Cleaning

The dataset was inspected for missing values and inconsistencies.

### Findings

* No missing values detected.
* No duplicate records requiring removal.
* All features were already numerical.

Result:

The dataset required minimal preprocessing.

---

# 5. Feature Engineering

Several preprocessing techniques were applied before model training.

### Steps Performed

#### Feature-Target Separation

Input Features:

```text
age, sex, cp, trestbps, chol,
fbs, restecg, thalach,
exang, oldpeak, slope,
ca, thal
```

Target:

```text
condition
```

#### Train-Test Split

The dataset was split into:

* Training Set: 80%
* Testing Set: 20%

Stratified sampling was used to preserve class distribution.

#### Feature Scaling

StandardScaler was applied to normalize feature values and improve model performance.

---

# 6. Handling Class Imbalance

Although the dataset was relatively balanced, class imbalance handling was included as part of the project requirements.

### Technique Used

SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic examples of the minority class, improving the model's ability to learn decision boundaries.

### Class Distribution

Before SMOTE:

```text
Condition 0 : 128
Condition 1 : 109
```

After SMOTE:

```text
Condition 0 : 128
Condition 1 : 128
```

---

# 7. Models Compared

Three machine learning algorithms were evaluated.

### Model 1: Logistic Regression

Advantages:

* Fast training
* Easy interpretation
* Strong baseline model

### Model 2: Random Forest

Advantages:

* Handles non-linear relationships
* Robust to overfitting
* Provides strong predictive performance

### Model 3: Gradient Boosting

Advantages:

* Captures complex patterns
* Strong ensemble learning capability

---

# 8. Cross Validation Results

Stratified 5-Fold Cross Validation was used to obtain reliable performance estimates.

| Model               | Mean F1 Score |
| ------------------- | ------------- |
| Logistic Regression | 0.8186        |
| Random Forest       | 0.8200        |
| Gradient Boosting   | 0.8092        |

### Best Performing Model

Random Forest achieved the highest mean F1 Score and was selected for further optimization.

---

# 9. Hyperparameter Tuning

RandomizedSearchCV was used to optimize Random Forest parameters.

### Search Space

```python
{
    "n_estimators":[100,200,300],
    "max_depth":[3,5,10,None],
    "min_samples_split":[2,5,10]
}
```

### Best Parameters

```python
{
    'n_estimators': 100,
    'min_samples_split': 5,
    'max_depth': 10
}
```

---

# 10. Final Model Evaluation

The tuned Random Forest model was evaluated on the unseen test dataset.

### Performance Metrics

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 86.67% |
| Precision | 91.67% |
| Recall    | 78.57% |
| F1 Score  | 84.62% |
| ROC-AUC   | 86.16% |

### Classification Report

| Class      | Precision | Recall | F1 Score |
| ---------- | --------- | ------ | -------- |
| No Disease | 0.83      | 0.94   | 0.88     |
| Disease    | 0.92      | 0.79   | 0.85     |

### Confusion Matrix

A confusion matrix was generated and saved as:

```text
confusion_matrix.png
```

---

# 11. Model Persistence

The final trained model and scaler were saved using Joblib.

Saved Files:

```text
models/
├── heart_disease_model.joblib
└── scaler.joblib
```

Benefits:

* Faster deployment
* Reusability
* Consistent predictions

---

# 12. Deployment

A Streamlit web application was developed to provide an interactive interface for users.

### Features

* User-friendly patient input form
* Real-time prediction
* Risk assessment display
* Model performance visualization
* EDA image visualization
* Confusion matrix display
* Model comparison results

### Run Application

```bash
py -m streamlit run app.py
```

---

# 13. Limitations

Despite achieving strong performance, several limitations remain.

### Limitations

* Small dataset size (297 records)
* Dataset collected from a single source
* No temporal patient history
* No external validation dataset
* Predictions should not replace professional medical diagnosis

---

# 14. Future Improvements

Several enhancements can improve the system further.

### Future Work

* Add XGBoost and LightGBM models
* Integrate SHAP explainability
* Deploy on Streamlit Cloud
* Build REST API using FastAPI
* Add feature importance visualizations
* Use larger healthcare datasets
* Support batch predictions

---

# 15. Conclusion

This project successfully implemented a complete machine learning pipeline for heart disease prediction using the UCI Cleveland Heart Disease Dataset.

The workflow included data analysis, preprocessing, class imbalance handling using SMOTE, model comparison through cross-validation, hyperparameter tuning, model evaluation, and deployment using Streamlit.

Among the evaluated algorithms, Random Forest achieved the best performance with an Accuracy of 86.67% and an F1 Score of 84.62%. The final system provides an effective demonstration of an end-to-end machine learning solution suitable for educational purposes, healthcare analytics projects, and internship portfolio development.
