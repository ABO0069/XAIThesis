import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt

# Import your local LLM
try:
    from llm_layer import ClinicalLLM
    LLM_AVAILABLE = True
except ImportError:
    st.error("Could not import 'llm_layer.py'. Make sure it is in the same folder.")
    LLM_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Diabetes Risk AI Assistant", layout="wide")

# --- CACHED RESOURCE LOADING (Prevents reloading on every click) ---
@st.cache_resource
def load_resources():
    """Loads models, scalers, and the LLM once."""
    try:
        # Load ML Artifacts
        rf_model = joblib.load('diabetes_rf_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        
        # Load Sample Data for LIME/SHAP initialization
        with open('sample_data.pkl', 'rb') as f:
            sample_data = pickle.load(f)
        X_train_sample = sample_data['X_train_sample']
        
        # Load Feature Names
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        feature_names = feature_info['feature_names']
        feature_cols = feature_info['feature_columns']

        # Initialize LIME Explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train_sample),
            feature_names=feature_names,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification'
        )

        # Initialize Local LLM (RTX 4080)
        clinical_llm = None
        if LLM_AVAILABLE:
            clinical_llm = ClinicalLLM()

        return rf_model, scaler, lime_explainer, X_train_sample, feature_names, feature_cols, clinical_llm

    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None, None, None

# Load everything
rf_model, scaler, lime_explainer, X_train_sample, feature_names, feature_cols, clinical_llm = load_resources()

# --- SIDEBAR: PATIENT DATA INPUT ---
st.sidebar.header("Patient Vitals & History")

def user_input_features():
    # Numerical Inputs
    age = st.sidebar.slider("Age", 0, 120, 45)
    pulse_rate = st.sidebar.slider("Pulse Rate (bpm)", 40, 180, 80)
    systolic_bp = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
    diastolic_bp = st.sidebar.slider("Diastolic BP (mmHg)", 40, 140, 80)
    glucose = st.sidebar.number_input("Glucose (mg/dL)", 50, 400, 100)
    height = st.sidebar.number_input("Height (cm)", 100, 250, 170)
    weight = st.sidebar.number_input("Weight (kg)", 30, 200, 75)
    
    # Calculated BMI
    bmi = weight / ((height/100) ** 2)
    st.sidebar.markdown(f"**Calculated BMI:** {bmi:.2f}")

    # Categorical Inputs (Mapped to 0/1)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender_encoded = 1 if gender == "Male" else 0
    
    fam_diabetes = st.sidebar.selectbox("Family History of Diabetes?", ["No", "Yes"])
    family_diabetes = 1 if fam_diabetes == "Yes" else 0
    
    hyp = st.sidebar.selectbox("Hypertensive?", ["No", "Yes"])
    hypertensive = 1 if hyp == "Yes" else 0
    
    fam_hyp = st.sidebar.selectbox("Family History of Hypertension?", ["No", "Yes"])
    family_hypertension = 1 if fam_hyp == "Yes" else 0
    
    cvd = st.sidebar.selectbox("Cardiovascular Disease?", ["No", "Yes"])
    cardiovascular_disease = 1 if cvd == "Yes" else 0
    
    stroke = st.sidebar.selectbox("History of Stroke?", ["No", "Yes"])
    stroke_hist = 1 if stroke == "Yes" else 0

    # Create Dictionary matching the training feature order EXACTLY
    data = {
        'age': age,
        'gender_encoded': gender_encoded,
        'pulse_rate': pulse_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'glucose': glucose,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'family_diabetes': family_diabetes,
        'hypertensive': hypertensive,
        'family_hypertension': family_hypertension,
        'cardiovascular_disease': cardiovascular_disease,
        'stroke': stroke_hist
    }
    
    # Convert to DataFrame with correct column order
    features = pd.DataFrame(data, index=[0])
    return features

# Get Input
input_df = user_input_features()

# --- MAIN DASHBOARD ---
st.title("🏥 AI Clinical Decision Support System")
st.markdown("Powered by **Random Forest + LIME + Local LLM (Phi-3)**")

# Display Input Data Summary
st.subheader("Patient Summary")
st.dataframe(input_df)

# --- ANALYZE BUTTON ---
if st.button("Run AI Analysis"):
    with st.spinner('Analyzing risk factors and generating clinical narrative...'):
        
        # 1. Prediction
        # Note: RF was trained on unscaled data in your pipeline, so we pass input_df directly
        prediction = rf_model.predict(input_df)[0]
        probability = rf_model.predict_proba(input_df)[0][1]
        
        # 2. LIME Explanation
        lime_exp = lime_explainer.explain_instance(
            input_df.values[0], 
            rf_model.predict_proba, 
            num_features=5
        )
        lime_features = lime_exp.as_list()
        
        # 3. LLM Narrative Generation
        patient_str = input_df.to_string(index=False)
        if clinical_llm:
            narrative = clinical_llm.generate_explanation(patient_str, probability, lime_features)
        else:
            narrative = "LLM not loaded."

    # --- DISPLAY RESULTS ---
    
    # Row 1: Risk Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Assessment")
        risk_color = "red" if probability > 0.5 else "green"
        st.markdown(f"## Probability: :{risk_color}[{probability*100:.1f}%]")
        if probability > 0.5:
            st.error("HIGH RISK DETECTED")
        else:
            st.success("LOW RISK DETECTED")
            
    with col2:
        st.subheader("Top Risk Drivers (LIME)")
        # Simple bar chart of LIME weights
        lime_data = pd.DataFrame(lime_features, columns=['Feature', 'Weight'])
        st.bar_chart(lime_data.set_index('Feature'))

    # Row 2: The AI Narrative
    st.markdown("---")
    st.header("📋 AI Clinical Report")
    st.info(narrative)

    # Row 3: Deep Dive (LIME Plot)
    st.markdown("---")
    with st.expander("View Detailed Explainability (LIME Plot)"):
        fig = lime_exp.as_pyplot_figure()
        st.pyplot(fig)