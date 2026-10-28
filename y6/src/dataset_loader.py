"""
dataset_loader.py - MediPredict AI
Loads and pre-processes datasets for:
1. Breast Cancer Wisconsin Diagnostic (from scikit-learn load_breast_cancer)
2. Heart Disease (UCI Cleveland dataset)
3. Diabetes (Pima Indians Diabetes dataset)
"""

import os
import urllib.request
import pandas as pd
from sklearn.datasets import load_breast_cancer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_file(urls, dest_path):
    """Attempts to download a file from a list of URLs with fallback."""
    for url in urls:
        try:
            print(f"Attempting download from: {url}")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            if os.path.getsize(dest_path) > 100:
                print(f"Successfully downloaded to {dest_path}")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    return False


def load_breast_cancer_data():
    """
    Loads Wisconsin Diagnostic Breast Cancer dataset from sklearn.
    Features: 10 mean nuclear characteristics.
    Target: 1 = Malignant, 0 = Benign.
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    
    # Sklearn target: 0 = malignant, 1 = benign.
    # Convert so 1 = Malignant (positive test), 0 = Benign for medical clarity
    df["target"] = (raw.target == 0).astype(int)
    
    mean_cols = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension"
    ]
    
    selected_df = df[mean_cols + ["target"]].copy()
    
    # Save local CSV for offline verification
    csv_path = os.path.join(DATA_DIR, "breast_cancer.csv")
    selected_df.to_csv(csv_path, index=False)
    
    X = selected_df[mean_cols]
    y = selected_df["target"]
    return X, y, mean_cols


def load_heart_disease_data():
    """
    Loads UCI Cleveland Heart Disease dataset.
    Target: 1 = Heart Disease Present, 0 = Normal/Absent.
    """
    csv_path = os.path.join(DATA_DIR, "heart.csv")
    
    if not os.path.exists(csv_path):
        urls = [
            "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
            "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/heart.csv",
            "https://archive.ics.uci.edu/static/public/45/data.csv"
        ]
        success = download_file(urls, csv_path)
        if not success:
            readme_path = os.path.join(DATA_DIR, "heart_README.txt")
            with open(readme_path, "w") as f:
                f.write(
                    "Heart Disease dataset missing.\n"
                    "Please download UCI heart.csv (Cleveland Heart Disease) and place it here as 'heart.csv'.\n"
                    "Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target\n"
                )
            print(f"[Notice] Heart Disease dataset unavailable. Instruction saved to {readme_path}")
            return None, None, None

    try:
        df = pd.read_csv(csv_path)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Standardize target column name
        if "num" in df.columns and "target" not in df.columns:
            df["target"] = df["num"]
        elif "output" in df.columns and "target" not in df.columns:
            df["target"] = df["output"]
            
        if "target" not in df.columns:
            print("[Error] 'target' column not found in heart.csv")
            return None, None, None

        # Ensure binary target (0 = no disease, 1 = disease)
        df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)
        
        feature_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        
        # Keep matching columns
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) < 5:
            # If columns differ, take all non-target numeric columns
            available_cols = [c for c in df.columns if c != "target"]
            
        X = df[available_cols].apply(pd.to_numeric, errors="coerce")
        y = df["target"]
        return X, y, available_cols
    except Exception as e:
        print(f"Error reading heart.csv: {e}")
        return None, None, None


def load_diabetes_data():
    """
    Loads Pima Indians Diabetes dataset.
    Target: 1 = Diabetic, 0 = Non-diabetic.
    """
    csv_path = os.path.join(DATA_DIR, "diabetes.csv")
    
    if not os.path.exists(csv_path):
        urls = [
            "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        ]
        success = download_file(urls, csv_path)
        if not success:
            readme_path = os.path.join(DATA_DIR, "diabetes_README.txt")
            with open(readme_path, "w") as f:
                f.write(
                    "Diabetes dataset missing.\n"
                    "Please download the Pima Indians Diabetes CSV and place it here as 'diabetes.csv'.\n"
                    "Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome\n"
                )
            print(f"[Notice] Diabetes dataset unavailable. Instruction saved to {readme_path}")
            return None, None, None

    try:
        df = pd.read_csv(csv_path)
        # Check if headerless
        if "outcome" not in [c.lower() for c in df.columns] and len(df.columns) == 9:
            df.columns = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
            ]
        
        df.columns = [c.strip() for c in df.columns]
        target_col = None
        for col in df.columns:
            if col.lower() in ["outcome", "target", "class"]:
                target_col = col
                break
                
        if not target_col:
            target_col = df.columns[-1]

        y = pd.to_numeric(df[target_col], errors="coerce").astype(int)
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        return X, y, feature_cols
    except Exception as e:
        print(f"Error reading diabetes.csv: {e}")
        return None, None, None

