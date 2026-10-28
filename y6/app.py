"""
app.py - MediPredict AI
College Machine Learning Project
Streamlit web application for medical disease prediction and model performance evaluation.
"""

import json
import os
import joblib
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Mandatory Educational Medical Disclaimer
DISCLAIMER_TEXT = (
    "⚠️ **EDUCATIONAL DISCLAIMER**: MediPredict AI is strictly an academic college machine learning project "
    "designed for demonstration and research purposes. It is **NOT** a certified medical diagnostic device and "
    "must **NOT** be used as a substitute for professional clinical advice, diagnosis, or treatment. "
    "Always consult a licensed healthcare professional for any medical concerns."
)


def render_disclaimer():
    """Renders the visible medical disclaimer."""
    st.warning(DISCLAIMER_TEXT)


@st.cache_resource
def load_bundle(dataset_name):
    """Loads saved joblib bundle for a dataset."""
    bundle_path = os.path.join(MODELS_DIR, f"{dataset_name}_models.joblib")
    if os.path.exists(bundle_path):
        try:
            return joblib.load(bundle_path)
        except Exception as e:
            st.error(f"Error loading bundle for {dataset_name}: {e}")
    return None


@st.cache_data
def load_all_metrics():
    """Loads metrics.json file."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error reading metrics: {e}")
    return None


def show_prediction_result(model, input_df, positive_label, negative_label):
    """Executes prediction using the actual trained pipeline and displays results."""
    try:
        prediction = model.predict(input_df)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
                prob = proba[1]  # Probability of positive class
            except Exception:
                pass

        st.markdown("---")
        st.subheader("Diagnostic Prediction Result")

        col1, col2 = st.columns([1, 1])

        with col1:
            if prediction == 1:
                st.error(f"🚨 **Assessment**: {positive_label}")
                st.markdown("The model identifies indicators consistent with high risk/presence of condition.")
            else:
                st.success(f"✅ **Assessment**: {negative_label}")
                st.markdown("The model identifies indicators consistent with low risk/absence of condition.")

        with col2:
            if prob is not None:
                st.metric(
                    label="Calculated Model Confidence (Positive Class Risk)",
                    value=f"{prob * 100:.1f}%"
                )
                st.progress(float(prob))
            else:
                st.info("Probability estimation is not available for this classifier configuration.")

        st.caption("Prediction generated directly by scikit-learn/XGBoost trained inference pipeline.")

    except Exception as e:
        st.error(f"Inference error: {e}")


# ==========================================
# PAGE 1: HOME & OVERVIEW
# ==========================================
def render_home():
    st.title("🩺 MediPredict AI")
    st.subheader("Multi-Disease Machine Learning Clinical Risk Assessment")
    render_disclaimer()

    st.markdown("""
    ### Project Overview
    **MediPredict AI** is a complete, modular machine learning system built for college-level healthcare analytics.
    It applies four supervised machine learning algorithms across three standard clinical benchmarks:
    
    1. **Heart Disease Risk Screening**: UCI Cleveland Heart Disease Dataset (13 clinical parameters)
    2. **Diabetes Risk Assessment**: Pima Indians Diabetes Dataset (8 clinical parameters)
    3. **Breast Cancer Diagnosis**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset (10 nuclear features)
    
    ### Algorithms Evaluated
    - **Logistic Regression**: Scaled linear probabilistic classification benchmark.
    - **Support Vector Machine (SVM)**: Scaled maximum-margin hyperplane classifier.
    - **Random Forest**: Non-linear ensemble of decision trees with bagging.
    - **XGBoost**: Extreme Gradient Boosting decision tree ensemble.
    
    ### Pipeline Architecture
    - **Stratified Train/Test Split (80/20)**: Preserves clinical class balance.
    - **Missing Value Imputation**: `SimpleImputer` with median strategy inside pipeline.
    - **Standard Scaling**: Applied to distance-sensitive models (Logistic Regression, SVM).
    - **Model Persistence**: Pipelines saved via `joblib` for reproducible offline inference.
    """)

    st.info("👈 Use the sidebar navigation on the left to test individual disease predictions or inspect Model Performance.")


# ==========================================
# PAGE 2: HEART DISEASE PREDICTION
# ==========================================
def render_heart_disease():
    st.title("❤️ Heart Disease Risk Prediction")
    st.markdown("Predict the likelihood of heart disease using 13 clinical biomarkers from the UCI Cleveland dataset.")
    render_disclaimer()

    bundle = load_bundle("heart_disease")
    if not bundle:
        st.error("Heart disease models are not trained yet. Run `python -m src.train` to train the models.")
        return

    models = bundle["models"]
    feature_names = bundle["feature_names"]
    best_model = bundle.get("best_model_name", list(models.keys())[0])

    # Model selector
    col_sel, col_preset = st.columns([2, 1])
    with col_sel:
        selected_model_name = st.selectbox(
            "Select Machine Learning Algorithm:",
            options=list(models.keys()),
            index=list(models.keys()).index(best_model) if best_model in models else 0,
            help=f"Defaulted to dataset's top performing model: {best_model}"
        )
    with col_preset:
        preset = st.radio("Quick Presets for Testing:", ["Custom", "High Risk Sample", "Low Risk Sample"], horizontal=True)

    # Preset values
    if preset == "High Risk Sample":
        d = {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3.0, "thal": 3.0}
    elif preset == "Low Risk Sample":
        d = {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0, "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 0.0, "thal": 3.0}
    else:
        d = {"age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0.0, "thal": 3.0}

    with st.form("heart_form"):
        st.subheader("Patient Clinical Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(d["age"]))
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], index=0 if d["sex"] == 1 else 1)[1]
            cp = st.selectbox(
                "Chest Pain Type",
                options=[
                    ("1: Typical Angina", 1),
                    ("2: Atypical Angina", 2),
                    ("3: Non-anginal Pain", 3),
                    ("4: Asymptomatic", 4)
                ],
                index=int(d["cp"]) - 1
            )[1]
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=int(d["trestbps"]))
            chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=int(d["chol"]))

        with c2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No (<= 120)", 0), ("Yes (> 120)", 1)], index=int(d["fbs"]))[1]
            restecg = st.selectbox(
                "Resting ECG Results",
                options=[
                    ("0: Normal", 0),
                    ("1: ST-T Wave Abnormality", 1),
                    ("2: Left Ventricular Hypertrophy", 2)
                ],
                index=int(d["restecg"])
            )[1]
            thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=int(d["thalach"]))
            exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], index=int(d["exang"]))[1]

        with c3:
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=8.0, value=float(d["oldpeak"]), step=0.1)
            slope = st.selectbox(
                "Peak Exercise ST Slope",
                options=[
                    ("1: Upsloping", 1),
                    ("2: Flat", 2),
                    ("3: Downsloping", 3)
                ],
                index=int(d["slope"]) - 1
            )[1]
            ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3], index=int(d["ca"]))
            thal = st.selectbox(
                "Thallium Stress Test Result",
                options=[
                    ("3: Normal", 3.0),
                    ("6: Fixed Defect", 6.0),
                    ("7: Reversible Defect", 7.0)
                ],
                index=0 if d["thal"] == 3.0 else (1 if d["thal"] == 6.0 else 2)
            )[1]

        submitted = st.form_submit_button("🩺 Predict Heart Disease Risk", use_container_width=True)

    if submitted:
        input_data = {
            "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps],
            "chol": [chol], "fbs": [fbs], "restecg": [restecg], "thalach": [thalach],
            "exang": [exang], "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal]
        }
        input_df = pd.DataFrame(input_data)[feature_names]
        show_prediction_result(
            model=models[selected_model_name],
            input_df=input_df,
            positive_label="High Risk of Heart Disease Detected",
            negative_label="Low Risk / No Significant Heart Disease Detected"
        )


# ==========================================
# PAGE 3: DIABETES PREDICTION
# ==========================================
def render_diabetes():
    st.title("🩸 Diabetes Risk Prediction")
    st.markdown("Predict the likelihood of diabetes using 8 clinical indicators from the Pima Indians Diabetes dataset.")
    render_disclaimer()

    bundle = load_bundle("diabetes")
    if not bundle:
        st.error("Diabetes models are not trained yet. Run `python -m src.train` to train the models.")
        return

    models = bundle["models"]
    feature_names = bundle["feature_names"]
    best_model = bundle.get("best_model_name", list(models.keys())[0])

    col_sel, col_preset = st.columns([2, 1])
    with col_sel:
        selected_model_name = st.selectbox(
            "Select Machine Learning Algorithm:",
            options=list(models.keys()),
            index=list(models.keys()).index(best_model) if best_model in models else 0,
            help=f"Defaulted to dataset's top performing model: {best_model}"
        )
    with col_preset:
        preset = st.radio("Quick Presets for Testing:", ["Custom", "Diabetic Sample", "Healthy Sample"], horizontal=True)

    if preset == "Diabetic Sample":
        d = {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 160, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}
    elif preset == "Healthy Sample":
        d = {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31}
    else:
        d = {"Pregnancies": 3, "Glucose": 115, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 28.5, "DiabetesPedigreeFunction": 0.450, "Age": 35}

    with st.form("diabetes_form"):
        st.subheader("Patient Clinical Measurements")
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, value=int(d["Pregnancies"]))
            glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", min_value=0, max_value=300, value=int(d["Glucose"]))
            bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=180, value=int(d["BloodPressure"]))
            skin = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=int(d["SkinThickness"]))

        with c2:
            insulin = st.number_input("2-Hour Serum Insulin (μU/mL)", min_value=0, max_value=900, value=int(d["Insulin"]))
            bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=70.0, value=float(d["BMI"]), step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.01, max_value=3.0, value=float(d["DiabetesPedigreeFunction"]), step=0.01)
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(d["Age"]))

        submitted = st.form_submit_button("🩸 Predict Diabetes Risk", use_container_width=True)

    if submitted:
        input_data = {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [bp],
            "SkinThickness": [skin],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age]
        }
        input_df = pd.DataFrame(input_data)[feature_names]
        show_prediction_result(
            model=models[selected_model_name],
            input_df=input_df,
            positive_label="High Risk of Diabetes Detected",
            negative_label="Low Risk / No Diabetes Detected"
        )


# ==========================================
# PAGE 4: BREAST CANCER PREDICTION
# ==========================================
def render_breast_cancer():
    st.title("🔬 Breast Cancer Diagnostic Prediction")
    st.markdown("Predict tumor malignancy from 10 nuclear features derived from fine-needle aspirate (FNA) images (Wisconsin Diagnostic Breast Cancer dataset).")
    render_disclaimer()

    bundle = load_bundle("breast_cancer")
    if not bundle:
        st.error("Breast cancer models are not trained yet. Run `python -m src.train` to train the models.")
        return

    models = bundle["models"]
    feature_names = bundle["feature_names"]
    best_model = bundle.get("best_model_name", list(models.keys())[0])

    col_sel, col_preset = st.columns([2, 1])
    with col_sel:
        selected_model_name = st.selectbox(
            "Select Machine Learning Algorithm:",
            options=list(models.keys()),
            index=list(models.keys()).index(best_model) if best_model in models else 0,
            help=f"Defaulted to dataset's top performing model: {best_model}"
        )
    with col_preset:
        preset = st.radio("Quick Presets for Testing:", ["Custom", "Malignant Sample", "Benign Sample"], horizontal=True)

    if preset == "Malignant Sample":
        d = {
            "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0,
            "mean smoothness": 0.1184, "mean compactness": 0.2776, "mean concavity": 0.3001,
            "mean concave points": 0.1471, "mean symmetry": 0.2419, "mean fractal dimension": 0.07871
        }
    elif preset == "Benign Sample":
        d = {
            "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46, "mean area": 566.3,
            "mean smoothness": 0.09779, "mean compactness": 0.08129, "mean concavity": 0.06664,
            "mean concave points": 0.04781, "mean symmetry": 0.1885, "mean fractal dimension": 0.05766
        }
    else:
        d = {
            "mean radius": 14.0, "mean texture": 18.0, "mean perimeter": 90.0, "mean area": 600.0,
            "mean smoothness": 0.10, "mean compactness": 0.10, "mean concavity": 0.08,
            "mean concave points": 0.05, "mean symmetry": 0.18, "mean fractal dimension": 0.06
        }

    with st.form("cancer_form"):
        st.subheader("Fine Needle Aspirate (FNA) Nuclear Characteristics")
        c1, c2 = st.columns(2)
        with c1:
            rad = st.number_input("Mean Radius (cell nucleus size)", min_value=5.0, max_value=35.0, value=float(d["mean radius"]), step=0.1)
            tex = st.number_input("Mean Texture (gray-scale variance)", min_value=5.0, max_value=45.0, value=float(d["mean texture"]), step=0.1)
            per = st.number_input("Mean Perimeter", min_value=40.0, max_value=250.0, value=float(d["mean perimeter"]), step=0.5)
            are = st.number_input("Mean Area", min_value=100.0, max_value=3000.0, value=float(d["mean area"]), step=1.0)
            smo = st.number_input("Mean Smoothness (local variation in radius lengths)", min_value=0.01, max_value=0.25, value=float(d["mean smoothness"]), step=0.005, format="%.5f")

        with c2:
            com = st.number_input("Mean Compactness (perimeter² / area - 1.0)", min_value=0.01, max_value=0.40, value=float(d["mean compactness"]), step=0.005, format="%.5f")
            con = st.number_input("Mean Concavity (severity of concave portions)", min_value=0.0, max_value=0.50, value=float(d["mean concavity"]), step=0.005, format="%.5f")
            cop = st.number_input("Mean Concave Points (number of concave portions)", min_value=0.0, max_value=0.25, value=float(d["mean concave points"]), step=0.005, format="%.5f")
            sym = st.number_input("Mean Symmetry", min_value=0.05, max_value=0.40, value=float(d["mean symmetry"]), step=0.005, format="%.5f")
            fra = st.number_input("Mean Fractal Dimension (coastline approximation - 1)", min_value=0.01, max_value=0.15, value=float(d["mean fractal dimension"]), step=0.005, format="%.5f")

        submitted = st.form_submit_button("🔬 Predict Breast Cancer Diagnosis", use_container_width=True)

    if submitted:
        input_data = {
            "mean radius": [rad], "mean texture": [tex], "mean perimeter": [per],
            "mean area": [are], "mean smoothness": [smo], "mean compactness": [com],
            "mean concavity": [con], "mean concave points": [cop], "mean symmetry": [sym],
            "mean fractal dimension": [fra]
        }
        input_df = pd.DataFrame(input_data)[feature_names]
        show_prediction_result(
            model=models[selected_model_name],
            input_df=input_df,
            positive_label="Malignant Tumor Indicated (High Risk)",
            negative_label="Benign Tumor Indicated (Low Risk)"
        )


# ==========================================
# PAGE 5: MODEL PERFORMANCE
# ==========================================
def render_model_performance():
    st.title("📊 Model Performance & Evaluation")
    st.markdown("Detailed quantitative benchmarks on held-out test splits (20% stratified test set).")
    render_disclaimer()

    all_metrics = load_all_metrics()
    if not all_metrics:
        st.error("No metrics found. Run `python -m src.train` to train the models.")
        return

    dataset_display_names = {
        "heart_disease": "UCI Cleveland Heart Disease",
        "diabetes": "Pima Indians Diabetes",
        "breast_cancer": "Wisconsin Diagnostic Breast Cancer"
    }

    selected_dataset = st.selectbox(
        "Select Dataset to Inspect:",
        options=list(all_metrics.keys()),
        format_func=lambda x: dataset_display_names.get(x, x)
    )

    dataset_metrics = all_metrics[selected_dataset]

    # Build summary comparison table
    table_rows = []
    for model_name, m in dataset_metrics.items():
        table_rows.append({
            "Algorithm": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1_score'] * 100:.2f}%",
            "Test Samples": m["test_samples"]
        })

    df_metrics = pd.DataFrame(table_rows)
    st.subheader(f"Performance Comparison — {dataset_display_names.get(selected_dataset, selected_dataset)}")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Highlight best model
    best_row = max(table_rows, key=lambda r: float(r["F1-Score"].replace("%", "")))
    st.success(f"🏆 **Top Performer by F1-Score**: **{best_row['Algorithm']}** with **{best_row['F1-Score']}** F1-Score and **{best_row['Accuracy']}** Accuracy.")

    st.markdown("---")
    st.subheader("Confusion Matrices (Test Split)")

    cols = st.columns(len(dataset_metrics))
    for idx, (model_name, m) in enumerate(dataset_metrics.items()):
        with cols[idx]:
            st.markdown(f"**{model_name}**")
            cm = m["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative (0)", "Actual Positive (1)"],
                columns=["Pred Negative (0)", "Pred Positive (1)"]
            )
            st.table(cm_df)
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            st.caption(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")


# ==========================================
# MAIN ROUTING
# ==========================================
def main():
    st.sidebar.title("🩺 MediPredict AI")
    st.sidebar.caption("College Machine Learning Diagnostic Suite")

    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "🏠 Overview",
            "❤️ Heart Disease Prediction",
            "🩸 Diabetes Prediction",
            "🔬 Breast Cancer Prediction",
            "📊 Model Performance"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    all_metrics = load_all_metrics()
    if all_metrics:
        st.sidebar.success(f"✅ Models Trained: {len(all_metrics)} Datasets")
    else:
        st.sidebar.warning("⚠️ Models Not Yet Trained")

    st.sidebar.markdown("---")
    st.sidebar.caption("MediPredict AI • Academic ML Project")

    if page == "🏠 Overview":
        render_home()
    elif page == "❤️ Heart Disease Prediction":
        render_heart_disease()
    elif page == "🩸 Diabetes Prediction":
        render_diabetes()
    elif page == "🔬 Breast Cancer Prediction":
        render_breast_cancer()
    elif page == "📊 Model Performance":
        render_model_performance()


if __name__ == "__main__":
    main()

