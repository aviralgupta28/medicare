# MediPredict AI 🩺

**MediPredict AI** is a complete, lightweight, and end-to-end college Machine Learning project built with **Python**, **scikit-learn**, **XGBoost**, and **Streamlit**. It provides multi-disease risk assessment across three benchmark clinical datasets:

1. **Heart Disease Risk Screening** (UCI Cleveland Heart Disease Dataset)
2. **Diabetes Risk Assessment** (Pima Indians Diabetes Dataset)
3. **Breast Cancer Diagnosis** (Wisconsin Diagnostic Breast Cancer Dataset)

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: MediPredict AI is developed solely for educational and academic demonstration purposes as a college machine learning project. It is **NOT** a certified medical device and must **NOT** be used for clinical diagnosis, medical treatment, or healthcare decision-making. Always consult a qualified medical professional for any health-related concerns.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have **Python 3.10+** (tested and verified on Python 3.14) installed on your system.

### 2. Install Dependencies
Open your terminal in the project root directory and run:
```bash
pip install -r requirements.txt
```
*(Or via python module: `python -m pip install -r requirements.txt`)*

### 3. Train the Models
Train the pipelines and evaluate test benchmarks:
```bash
python -m src.train
```
This automatically:
- Loads the datasets (downloads/caches them if missing).
- Splits data with stratified 80/20 train/test splits.
- Builds preprocessing pipelines (`SimpleImputer`, `StandardScaler`).
- Trains **Logistic Regression**, **Support Vector Machine (SVM)**, **Random Forest**, and **XGBoost**.
- Computes genuine test metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Serializes trained pipelines to `models/*.joblib` and metrics to `models/metrics.json`.

### 4. Launch the Streamlit Web Application
Run:
```bash
streamlit run app.py
```
*(Or via python module: `python -m streamlit run app.py`)*

Once started, open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
MediPredict-AI/
├── data/                       # Cached and local benchmark datasets
│   ├── heart.csv               # UCI Cleveland Heart Disease dataset
│   ├── diabetes.csv            # Pima Indians Diabetes dataset
│   └── breast_cancer.csv       # Wisconsin Diagnostic Breast Cancer dataset
├── models/                     # Saved model artifacts & metrics
│   ├── heart_disease_models.joblib
│   ├── diabetes_models.joblib
│   ├── breast_cancer_models.joblib
│   └── metrics.json            # Actual test evaluation metrics
├── src/                        # Source logic
│   ├── __init__.py
│   ├── dataset_loader.py       # Data loading, cleaning & schema validation
│   └── train.py                # Pipeline construction, training & evaluation
├── app.py                      # Multi-page Streamlit web application
├── requirements.txt            # Minimal pip dependencies
└── README.md                   # Project documentation & instructions
```

---

## 🧪 Actual Model Performance Benchmarks

*All metrics are evaluated strictly on held-out stratified 20% test splits. No fabricated metrics.*

### 1. Heart Disease (UCI Cleveland — 303 samples, 13 features)
| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Top)** | **88.52%** | **83.87%** | **92.86%** | **88.14%** |
| Logistic Regression | 86.89% | 81.25% | 92.86% | 86.67% |
| XGBoost | 85.25% | 78.79% | 92.86% | 85.25% |
| Support Vector Machine | 85.25% | 80.65% | 89.29% | 84.75% |

### 2. Breast Cancer (Wisconsin Diagnostic — 569 samples, 10 features)
| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Top)** | **95.61%** | **95.12%** | **92.86%** | **93.98%** |
| Support Vector Machine | 95.61% | 97.44% | 90.48% | 93.83% |
| Random Forest | 93.86% | 90.70% | 92.86% | 91.76% |
| Logistic Regression | 92.98% | 88.64% | 92.86% | 90.70% |

### 3. Diabetes (Pima Indians — 768 samples, 8 features)
| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (Top)** | **75.32%** | **66.00%** | **61.11%** | **63.46%** |
| Random Forest | 75.97% | 68.09% | 59.26% | 63.37% |
| XGBoost | 74.03% | 64.00% | 59.26% | 61.54% |
| Logistic Regression | 71.43% | 60.87% | 51.85% | 56.00% |

---

## ⚙️ Offline / Manual Dataset Setup (Fallback)

If internet access is restricted or offline:
- **Heart Disease**: Place the UCI Cleveland CSV file at `data/heart.csv` with columns: `age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num`.
- **Diabetes**: Place the Pima Indians Diabetes CSV file at `data/diabetes.csv` with columns: `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`.
- **Breast Cancer**: Always automatically loaded and extracted locally from `sklearn.datasets.load_breast_cancer`.

---

## 🎓 Academic Presentation Highlights
- **End-to-End Pipeline Integrity**: Preprocessing steps (`SimpleImputer` + `StandardScaler`) are encapsulated in scikit-learn `Pipeline` objects to prevent data leakage during train/test splits.
- **Model Comparison**: Allows switching between 4 distinct classifiers in real-time during interactive inference.
- **Confidence Scoring**: Outputs predicted positive-class risk probability alongside categorical classification.
- **Reproducibility**: Fixed random seeds (`random_state=42`) ensure consistent metrics across runs.

