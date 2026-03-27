# 📊 Telco Customer Churn Prediction

---

## 📌 Overview

This project builds a machine learning pipeline to predict **customer churn** using the Telco Customer Churn dataset.

The model helps identify customers likely to leave a telecom service, enabling businesses to take proactive retention strategies.

A **Random Forest Classifier** is used along with preprocessing and hyperparameter tuning for optimal performance.

---

## ⚙️ Features

✔ Data cleaning and preprocessing
✔ Missing value handling
✔ Feature scaling & encoding
✔ Pipeline-based model building
✔ Hyperparameter tuning (GridSearchCV)
✔ Model evaluation (Accuracy + Classification Report)
✔ Model saving using Joblib

---

## 📂 Dataset

Dataset used: **Telco Customer Churn Dataset**

### 🎯 Target Variable

* `Churn`

  * Yes → 1
  * No → 0

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**

---

## 🔄 Workflow

### 1️⃣ Data Preprocessing

* Converted `TotalCharges` to numeric
* Handled missing values using median
* Dropped `customerID` column
* Encoded target variable

---

### 2️⃣ Feature Engineering

**Numerical Features:**

* tenure
* MonthlyCharges
* TotalCharges

**Categorical Features:**

* Remaining columns

---

### 3️⃣ Pipeline

```python
Pipeline([
  ('preprocessor', ColumnTransformer(...)),
  ('classifier', RandomForestClassifier())
])
```

---

### 4️⃣ Hyperparameter Tuning

```python
param_grid = {
  'classifier__n_estimators': [100, 200],
  'classifier__max_depth': [5, 10, None]
}
```

* Used **GridSearchCV**
* 3-fold cross-validation

---

### 5️⃣ Model Evaluation

* Accuracy Score
* Precision
* Recall
* F1-score

---

### 6️⃣ Model Saving

```python
joblib.dump(model, 'telco_churn_pipeline.pkl')
```

---

## 📈 Results

* Best parameters selected automatically
* Evaluated on unseen test data

Example outputs:

* ✅ Best Parameters
* ✅ Accuracy Score
* ✅ Classification Report

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install pandas numpy scikit-learn joblib
```

---

### ▶️ Run the Project

```bash
python churn_model.py
```

---

### 📁 Update Dataset Path

```python
filepath = "your_dataset_path.csv"
```

---

## 💾 Model Usage

```python
import joblib

model = joblib.load('telco_churn_pipeline.pkl')
predictions = model.predict(new_data)
```

---

## 📌 Future Improvements

* 🚀 Try XGBoost / LightGBM
* 📊 Feature selection
* ⚖️ Handle class imbalance
* 🌐 Deploy with Flask / FastAPI



