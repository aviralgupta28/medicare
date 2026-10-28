"""
train.py - MediPredict AI
Trains machine learning pipelines (Logistic Regression, SVM, Random Forest, XGBoost)
for Heart Disease, Diabetes, and Breast Cancer datasets with:
- Stratified train/test split
- SimpleImputer for missing-value handling
- StandardScaler for Logistic Regression & SVM
- Actual test metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix
- Model persistence using joblib
"""

import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Check XGBoost availability
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[Warning] XGBoost is not available. Continuing with Logistic Regression, SVM, and Random Forest.")

from src.dataset_loader import (
    load_breast_cancer_data,
    load_diabetes_data,
    load_heart_disease_data,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_models():
    """Builds model dictionary with proper preprocessing pipelines."""
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42))
        ])

    return models


def train_and_evaluate_dataset(dataset_name, X, y, feature_names):
    """
    Trains all models on a given dataset using a stratified split,
    computes test metrics, and saves the trained pipeline bundle.
    """
    print(f"\n==========================================")
    print(f"  Training models for: {dataset_name.upper()}")
    print(f"==========================================")
    print(f"Total samples: {len(X)}, Features: {len(feature_names)}")

    # Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models_dict = build_models()
    trained_pipelines = {}
    metrics_summary = {}

    best_model_name = None
    best_f1 = -1.0

    for model_name, pipeline in models_dict.items():
        print(f"-> Training {model_name}...")
        pipeline.fit(X_train, y_train)

        # Actual test metrics
        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        trained_pipelines[model_name] = pipeline
        metrics_summary[model_name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "test_samples": int(len(y_test))
        }

        print(f"   Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name

    print(f"Best performing model for {dataset_name}: {best_model_name} (F1: {best_f1:.4f})")

    # Save trained pipelines with joblib
    save_payload = {
        "dataset_name": dataset_name,
        "feature_names": feature_names,
        "models": trained_pipelines,
        "best_model_name": best_model_name,
        "metrics": metrics_summary
    }
    model_path = os.path.join(MODELS_DIR, f"{dataset_name}_models.joblib")
    joblib.dump(save_payload, model_path)
    print(f"Saved pipeline bundle to: {model_path}")

    return metrics_summary


def train_all():
    """Main training orchestration function."""
    datasets = {
        "heart_disease": load_heart_disease_data,
        "diabetes": load_diabetes_data,
        "breast_cancer": load_breast_cancer_data
    }

    all_metrics = {}

    for name, loader in datasets.items():
        X, y, feature_names = loader()
        if X is None or y is None:
            print(f"[Skipping] {name} dataset could not be loaded. Continuing with remaining datasets.")
            continue
        metrics = train_and_evaluate_dataset(name, X, y, feature_names)
        all_metrics[name] = metrics

    # Save overall metrics to JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics successfully saved to {metrics_path}")
    print("Training phase complete!")


if __name__ == "__main__":
    train_all()

