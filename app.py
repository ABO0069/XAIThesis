"""
Simplified Diabetes Risk Assessment System
===========================================
Clean, user-friendly interface for patients and healthcare providers
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import lime.lime_tabular
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')

try:
    from llm_layer import ClinicalLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

st.set_page_config(
    page_title="Diabetes Risk Assessment", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Feature columns
FEATURE_COLUMNS = [
    'age', 'gender_encoded', 'pulse_rate', 'systolic_bp', 'diastolic_bp',
    'glucose', 'height', 'weight', 'bmi', 'family_diabetes', 'hypertensive',
    'family_hypertension', 'cardiovascular_disease', 'stroke'
]

# Load models and resources
@st.cache_resource
def load_resources():
    """Load all necessary models and data"""
    try:
        # Load models (use XGBoost as primary for best performance)
        rf_model = joblib.load('diabetes_rf_model.pkl')
        
        try:
            xgb_model = joblib.load('diabetes_xgb_model.pkl')
            primary_model = xgb_model  # Use XGBoost for predictions
        except:
            primary_model = rf_model
        
        # Load other resources
        with open('sample_data.pkl', 'rb') as f:
            sample_data = pickle.load(f)
        X_train_sample = sample_data['X_train_sample']
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        feature_names = feature_info['feature_names']
        
        # LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train_sample),
            feature_names=feature_names,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification',
            discretize_continuous=True
        )
        
        # LLM
        clinical_llm = None
        if LLM_AVAILABLE:
            try:
                clinical_llm = ClinicalLLM()
            except:
                pass
        
        return primary_model, rf_model, lime_explainer, feature_names, clinical_llm
    
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None

# Load everything
primary_model, rf_model, lime_explainer, feature_names, clinical_llm = load_resources()

# Sidebar - Simple input form (moved before header so patient_name is defined)
st.sidebar.header("📋 Enter Your Information")

with st.sidebar:
    st.markdown("### Your Name")
    patient_name = st.text_input("Full Name", value="", placeholder="e.g., John Smith")
    
    if patient_name:
        st.success(f"Welcome, {patient_name}! 👋")

# Header (uses patient_name from sidebar)
if patient_name:
    st.markdown(f'<p class="main-header">🏥 Diabetes Risk Assessment for {patient_name}</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="main-header">🏥 Diabetes Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Health Screening • Powered by Microsoft Phi-3 Mini AI</p>', unsafe_allow_html=True)

# Continue with rest of sidebar
with st.sidebar:
    st.markdown("### Personal Details")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
    
    st.markdown("### Vital Signs")
    
    col1, col2 = st.columns(2)
    with col1:
        pulse = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=200, value=75, step=1)
    with col2:
        glucose = st.number_input("Glucose (mmol/L)", min_value=3.0, max_value=30.0, value=5.5, step=0.1)
    
    st.caption("💡 Normal fasting: 3.9-5.5 • Pre-diabetes: 5.6-6.9 • Diabetes: ≥7.0")
    
    col1, col2 = st.columns(2)
    with col1:
        systolic = st.number_input("Blood Pressure (Top)", min_value=80, max_value=220, value=120, step=1)
    with col2:
        diastolic = st.number_input("Blood Pressure (Bottom)", min_value=40, max_value=150, value=80, step=1)
    
    st.caption(f"Your BP: {systolic}/{diastolic} mmHg")
    
    st.markdown("### Physical Measurements")
    
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
    
    bmi = weight / (height ** 2)
    
    # BMI indicator
    if bmi < 18.5:
        bmi_status = "Underweight"
        bmi_color = "blue"
    elif bmi < 25:
        bmi_status = "Normal"
        bmi_color = "green"
    elif bmi < 30:
        bmi_status = "Overweight"
        bmi_color = "orange"
    else:
        bmi_status = "Obese"
        bmi_color = "red"
    
    st.markdown(f"**BMI:** {bmi:.1f} — :{bmi_color}[{bmi_status}]")
    
    st.markdown("### Medical History")
    
    family_diabetes = st.checkbox("Family history of diabetes")
    hypertensive = st.checkbox("Currently have high blood pressure")
    family_hypertension = st.checkbox("Family history of high blood pressure")
    cvd = st.checkbox("Heart disease")
    stroke = st.checkbox("Previous stroke")

# Create patient data
gender_encoded = 1 if gender == "Male" else 0

patient_data = {
    'age': age,
    'gender_encoded': gender_encoded,
    'pulse_rate': pulse,
    'systolic_bp': systolic,
    'diastolic_bp': diastolic,
    'glucose': glucose,
    'height': height,
    'weight': weight,
    'bmi': bmi,
    'family_diabetes': 1 if family_diabetes else 0,
    'hypertensive': 1 if hypertensive else 0,
    'family_hypertension': 1 if family_hypertension else 0,
    'cardiovascular_disease': 1 if cvd else 0,
    'stroke': 1 if stroke else 0
}

input_df = pd.DataFrame([patient_data])

# Main content area
st.markdown("---")

# Analyze button
if st.button("🔬 Analyze My Risk", type="primary", use_container_width=True):
    
    with st.spinner('Analyzing your health data with AI...'):
        
        # Get prediction
        prediction = primary_model.predict(input_df)[0]
        probability = primary_model.predict_proba(input_df)[0][1]
        
        # LIME explanation
        lime_exp = lime_explainer.explain_instance(
            input_df.values[0],
            rf_model.predict_proba,
            num_features=5
        )
        lime_features = lime_exp.as_list(label=1)
        
        # Generate AI narratives
        patient_str = input_df.to_string(index=False)
        
        if clinical_llm and clinical_llm.model:
            try:
                patient_narrative = clinical_llm.generate_patient_narrative(
                    patient_str, probability, lime_features, None, patient_name
                )
                doctor_report = clinical_llm.generate_doctor_report(
                    patient_str, probability, lime_features, None, patient_name
                )
            except:
                patient_narrative = None
                doctor_report = None
        else:
            patient_narrative = None
            doctor_report = None
    
    # Display Results
    st.markdown("---")
    
    # Risk Assessment (Big and clear)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        risk_percent = probability * 100
        
        if probability >= 0.7:
            risk_level = "Very High Risk"
            risk_color = "🔴"
            alert_type = "error"
        elif probability >= 0.5:
            risk_level = "High Risk"
            risk_color = "🟠"
            alert_type = "warning"
        elif probability >= 0.3:
            risk_level = "Moderate Risk"
            risk_color = "🟡"
            alert_type = "info"
        else:
            risk_level = "Low Risk"
            risk_color = "🟢"
            alert_type = "success"
        
        st.markdown(f"### {risk_color} Diabetes Risk Assessment")
        st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>{risk_percent:.1f}%</h1>", 
                   unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{risk_level}</h3>", 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Factors Visualization (Simple, color-coded)
    st.markdown("### 📊 Key Health Factors")
    
    # Separate positive and negative factors
    risk_increasers = [(feat, weight) for feat, weight in lime_features if weight > 0]
    risk_decreasers = [(feat, weight) for feat, weight in lime_features if weight < 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Risk Factors")
        if risk_increasers:
            for feat, weight in risk_increasers:
                # Clean up feature name
                feat_clean = feat.split('<')[0].split('>')[0].split('=')[0].strip()
                st.markdown(f"- **{feat_clean}**")
        else:
            st.markdown("*None identified*")
    
    with col2:
        st.markdown("#### 🟢 Protective Factors")
        if risk_decreasers:
            for feat, weight in risk_decreasers:
                feat_clean = feat.split('<')[0].split('>')[0].split('=')[0].strip()
                st.markdown(f"- **{feat_clean}**")
        else:
            st.markdown("*None identified*")
    
    # Simple bar chart
    st.markdown("### 📈 Factor Impact Visualization")
    
    feature_names_clean = []
    feature_weights = []
    feature_colors = []
    
    for feat, weight in lime_features[:5]:
        # Clean feature name
        feat_clean = feat.split('<')[0].split('>')[0].split('=')[0].strip()
        feature_names_clean.append(feat_clean)
        feature_weights.append(abs(weight))
        feature_colors.append('#ff4444' if weight > 0 else '#44ff44')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(feature_names_clean, feature_weights, color=feature_colors)
    ax.set_xlabel('Impact on Risk', fontsize=12)
    ax.set_title('Top 5 Factors Affecting Your Risk', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff4444', label='Increases Risk'),
        Patch(facecolor='#44ff44', label='Decreases Risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # AI-Generated Explanations (Tabs for patient and doctor)
    if patient_narrative or doctor_report:
        st.markdown("### 🤖 AI Health Assistant (Powered by Microsoft Phi-3)")
        
        tab1, tab2 = st.tabs(["💬 For You", "👨‍⚕️ For Healthcare Provider"])
        
        with tab1:
            if patient_narrative:
                st.info(patient_narrative)
            else:
                st.info("Your diabetes risk assessment is complete. Please consult with a healthcare provider for personalized advice.")
        
        with tab2:
            if doctor_report:
                st.markdown(doctor_report)
            else:
                st.markdown(f"**Risk Level:** {risk_level} ({risk_percent:.1f}%)\n\n"
                          f"**Key Factors:** {', '.join([f.split('<')[0].split('>')[0].strip() for f, w in lime_features[:3]])}")
    
    else:
        # Fallback if LLM not available
        st.markdown("### 📋 What This Means")
        
        if probability >= 0.5:
            st.warning(f"⚠️ Your assessment indicates {risk_level.lower()}. We strongly recommend:\n"
                      f"- Schedule an appointment with your doctor\n"
                      f"- Get comprehensive diabetes testing\n"
                      f"- Discuss lifestyle modifications")
        else:
            st.success(f"✅ Your assessment shows {risk_level.lower()}. Continue:\n"
                      f"- Maintaining healthy lifestyle\n"
                      f"- Regular health checkups\n"
                      f"- Monitoring key health metrics")

else:
    # Initial state
    st.info("👈 Enter your health information in the sidebar, then click **'Analyze My Risk'** to get your assessment.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Accurate")
        st.markdown("Advanced AI models analyze 14 health factors to assess your diabetes risk with high precision.")
    
    with col2:
        st.markdown("### 🔒 Private")
        st.markdown("Your data stays on this device. No information is stored or shared.")
    
    with col3:
        st.markdown("### ⚡ Instant")
        st.markdown("Get your risk assessment and personalized insights in seconds.")

# Footer
st.markdown("---")
st.caption("⚕️ **Medical Disclaimer:** This is a screening tool, not a diagnosis. Always consult healthcare professionals for medical decisions.")

if torch.cuda.is_available():
    st.caption(f"🚀 Powered by AI running on {torch.cuda.get_device_name(0)}")
else:
    st.caption("💻 Powered by AI")