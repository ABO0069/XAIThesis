"""
Enhanced Clinical LLM Layer with Counterfactual Integration
============================================================
Generates patient-friendly narratives that include:
1. Risk assessment
2. LIME explanations (why you're at risk)
3. Counterfactual interventions (what you can do about it)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

class ClinicalLLM:
    def __init__(self):
        print(f"Loading {MODEL_ID} to RTX 4080...")
        try:
            # Check GPU availability first
            if not torch.cuda.is_available():
                print("⚠️  WARNING: No GPU detected! LLM will be very slow on CPU.")
                print("    Please check CUDA installation and GPU drivers.")
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✓ GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, 
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 2. Load Model - FORCE GPU usage
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Changed from torch_dtype
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                attn_implementation="eager"
            )
            
            # Verify model is on GPU
            device_used = next(self.model.parameters()).device
            print(f"Model loaded on device: {device_used}")
            
            # 3. Create Pipeline (FIXED: removed device parameter, increased tokens)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=800,  # Increased from 450 to prevent truncation
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
                # NOTE: Do NOT specify device when using device_map in model loading
            )
            
            # System prompts for different audiences
            self.doctor_prompt = """You are a Clinical Decision Support System (CDSS) Assistant. 
Interpret diabetes risk predictions based STRICTLY on the provided data. 
Use professional medical language for healthcare providers."""
            
            self.patient_prompt = """You are a helpful medical assistant explaining diabetes risk to patients.
Use simple, clear language. Be empathetic and encouraging. 
Focus on actionable advice without medical jargon."""
            
            # Final confirmation
            if torch.cuda.is_available() and device_used.type == 'cuda':
                print("Device set to use cuda")
                print("SUCCESS: Local LLM loaded on GPU!")
            else:
                print("Device set to use cpu")
                print("SUCCESS: Local LLM loaded on CPU (will be slower)")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading LLM: {e}")
            print("Pipeline will continue but LLM narratives will show error messages.")
            self.model = None
            self.pipe = None
    
    def generate_explanation(self, patient_data_str, probability, lime_features):
        """Original function for doctor-focused explanations (kept for compatibility)"""
        if not self.model:
            return "Error: LLM model not loaded. Please check GPU availability and model installation."
        
        risk_str = f"{probability*100:.1f}%"
        risk_level = "HIGH" if probability > 0.5 else "LOW"
        
        # Handle different LIME feature formats
        features_text = ""
        if isinstance(lime_features, list):
            if len(lime_features) > 0 and isinstance(lime_features[0], tuple):
                features_text = "\n".join([f"- {feat}: {weight:.4f}" for feat, weight in lime_features[:5]])
            else:
                features_text = str(lime_features)
        else:
            features_text = str(lime_features)
        
        messages = [
            {"role": "system", "content": self.doctor_prompt},
            {"role": "user", "content": f"""
**Patient Data:**
{patient_data_str}

**Risk Assessment:**
- Prediction: {risk_level} ({risk_str})

**Key Drivers (LIME Analysis):**
{features_text}

Write a concise clinical summary for a doctor explaining this patient's diabetes risk.
            """}
        ]
        
        try:
            outputs = self.pipe(messages, use_cache=False)
            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def generate_patient_narrative(self, patient_data_str, probability, lime_features, 
                                   counterfactual_interventions=None, patient_name=None):
        """
        Generate patient-friendly narrative with counterfactual interventions.
        
        Args:
            patient_data_str: Patient demographics and measurements
            probability: Diabetes risk probability (0-1)
            lime_features: List of (feature, importance) tuples
            counterfactual_interventions: List of dicts with intervention details
            patient_name: Patient's name for personalization (optional)
        """
        if not self.model:
            return "Error: LLM model not loaded. Please check GPU availability and model installation."
        
        risk_str = f"{probability*100:.1f}%"
        risk_level = "High" if probability > 0.5 else "Low"
        
        # Personalized greeting
        greeting = f"Hello {patient_name}, " if patient_name else ""
        
        # Format LIME features in simple language
        risk_factors = []
        try:
            if isinstance(lime_features, list):
                for feat, weight in lime_features[:5]:  # Top 5 features
                    if abs(weight) > 0.05:
                        impact = "increases" if weight > 0 else "decreases"
                        # Clean feature name
                        feat_clean = str(feat).split('<')[0].split('>')[0].strip()
                        risk_factors.append(f"- Your {feat_clean.lower()} {impact} your risk")
        except Exception as e:
            risk_factors.append("- Multiple factors contribute to your risk")
        
        features_text = "\n".join(risk_factors) if risk_factors else "- Multiple factors contribute to your risk"
        
        # Format counterfactual interventions
        interventions_text = ""
        if counterfactual_interventions and len(counterfactual_interventions) > 0:
            interventions_text = "\n**What You Can Do (Recommended Changes):**\n"
            for i, intervention in enumerate(counterfactual_interventions[:3], 1):  # Top 3
                try:
                    interventions_text += f"\n{i}. **{intervention.get('feature', 'Health Parameter')}**\n"
                    interventions_text += f"   - Current: {intervention.get('current', 0):.1f}\n"
                    interventions_text += f"   - Target: {intervention.get('target', 0):.1f} ({intervention.get('change_pct', 0):+.0f}%)\n"
                    interventions_text += f"   - How: {intervention.get('intervention', 'Lifestyle modification')}\n"
                    
                    if intervention.get('methods'):
                        methods = intervention['methods']
                        if isinstance(methods, list):
                            interventions_text += f"   - Steps: {', '.join(methods[:2])}\n"
                        else:
                            interventions_text += f"   - Steps: {methods}\n"
                    
                    interventions_text += f"   - Timeframe: {intervention.get('timeframe', '3-6 months')}\n"
                except Exception as e:
                    continue  # Skip malformed interventions
        
        # Create patient-friendly prompt with name
        if patient_name:
            name_instruction = f"Address the patient as '{patient_name}' at the start of your message. Use their name naturally in your greeting."
            greeting_example = f"Start with 'Hello {patient_name}' or 'Hi {patient_name}'. End with a friendly closing like 'Take care' or 'Best wishes' - do NOT sign with the patient's name."
        else:
            name_instruction = "Address the patient in a friendly, professional way."
            greeting_example = "Use a warm, professional tone. End with 'Take care' or 'Best wishes'."
        
        messages = [
            {"role": "system", "content": self.patient_prompt + "\n\nIMPORTANT: Never use placeholders like [Name] or [Your Name]. When greeting, use the actual patient name. When signing off, use phrases like 'Take care', 'Best wishes', or 'Stay healthy' - never sign with the patient's own name."},
            {"role": "user", "content": f"""
**Patient Information:**
{patient_data_str}
{f"**Patient Name: {patient_name}**" if patient_name else ""}

**Your Diabetes Risk:** {risk_level} ({risk_str})

**Why You're At Risk:**
{features_text}
{interventions_text}

Write a clear, encouraging message for the patient (under 120 words):
1. Their risk level in simple terms
2. Main contributing factors
3. Specific actions they can take

{name_instruction}
{greeting_example}

CRITICAL: 
- GREET with the actual name "{patient_name}" (e.g., "Hello {patient_name}")
- NEVER use placeholders like "[Your Name]" or "[Name]"
- SIGN OFF with "Take care" or "Best wishes" - NOT with the patient's name
Be positive and actionable. No medical jargon.
            """}
        ]
        
        try:
            outputs = self.pipe(messages, use_cache=False)
            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            return f"Error generating narrative: {str(e)}"
    
    def generate_doctor_report(self, patient_data_str, probability, lime_features,
                              counterfactual_interventions=None, patient_name=None):
        """
        Generate detailed clinical report with counterfactual pathways.
        For healthcare providers with technical details.
        
        Args:
            patient_name: Patient's name for the report (optional)
        """
        if not self.model:
            return "Error: LLM model not loaded. Please check GPU availability and model installation."
        
        risk_str = f"{probability*100:.1f}%"
        # More nuanced risk levels
        if probability >= 0.7:
            risk_level = "VERY HIGH RISK"
        elif probability >= 0.5:
            risk_level = "HIGH RISK"
        elif probability >= 0.3:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK"
        
        # Patient identifier for report
        patient_id = f"Patient: {patient_name}" if patient_name else "Patient Assessment"
        
        # Format LIME features
        features_text = ""
        try:
            if isinstance(lime_features, list):
                features_text = "\n".join([f"- {feat}: {weight:+.4f}" for feat, weight in lime_features[:8]])
            else:
                features_text = str(lime_features)
        except:
            features_text = "- Feature attribution data unavailable"
        
        # Format counterfactuals with clinical detail
        interventions_text = ""
        if counterfactual_interventions and len(counterfactual_interventions) > 0:
            interventions_text = "\n**Clinical Intervention Pathways (DiCE Analysis):**\n"
            for i, cf in enumerate(counterfactual_interventions[:3], 1):  # Top 3
                try:
                    interventions_text += f"\nPathway {i} - {cf.get('priority', 'Priority 2')}:\n"
                    interventions_text += f"  Target: {cf.get('feature', 'Parameter')}\n"
                    interventions_text += f"  Change: {cf.get('current', 0):.1f} → {cf.get('target', 0):.1f} ({cf.get('change_pct', 0):+.0f}%)\n"
                    interventions_text += f"  Intervention: {cf.get('intervention', 'Lifestyle modification')}\n"
                    interventions_text += f"  Feasibility: {cf.get('feasibility', 'Moderate')}\n"
                    interventions_text += f"  Clinical Impact: {cf.get('impact', 'High')}\n"
                    interventions_text += f"  Timeframe: {cf.get('timeframe', '3-6 months')}\n"
                    
                    if cf.get('methods'):
                        methods = cf['methods']
                        if isinstance(methods, list):
                            interventions_text += f"  Methods:\n"
                            for method in methods[:3]:
                                interventions_text += f"    - {method}\n"
                except Exception as e:
                    continue  # Skip malformed interventions
        
        messages = [
            {"role": "system", "content": self.doctor_prompt},
            {"role": "user", "content": f"""
**{patient_id}**
{patient_data_str}

**Risk Assessment:** {risk_level} ({risk_str})

**Feature Attribution (LIME):**
{features_text}
{interventions_text}

Generate a clinical decision support report with these 3 sections:
1. Risk Stratification (2 sentences max)
2. Key Risk Drivers (2 sentences max)  
3. Recommended Intervention Strategy (3 sentences max)

{f"Refer to the patient as {patient_name} in the report." if patient_name else ""}
Keep total under 150 words. Be concise and actionable.
            """}
        ]
        
        try:
            outputs = self.pipe(messages, use_cache=False)
            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            return f"Error generating report: {str(e)}"


# ============================================================================
# HELPER FUNCTION: Extract Counterfactuals from DiCE Output
# ============================================================================

def extract_counterfactual_interventions(cf_analysis_sorted, original_patient):
    """
    Convert counterfactual analysis into LLM-friendly format.
    
    Args:
        cf_analysis_sorted: List of counterfactual dicts from your analysis
        original_patient: Original patient data (Series or dict)
    
    Returns:
        List of intervention dicts for LLM
    """
    interventions = []
    
    # Get the best counterfactual pathway (Priority 1)
    if cf_analysis_sorted and len(cf_analysis_sorted) > 0:
        best_cf = cf_analysis_sorted[0]
        
        changes = best_cf.get('Changes', [])
        if not changes:
            return []
        
        for change in changes:
            try:
                # Parse the change string (e.g., "Glucose: 12.5 → 8.8 (-30%)")
                feature = change.get('feature', '')
                change_detail = change.get('change', '')
                
                # Extract current and target values
                if '→' in change_detail:
                    parts = change_detail.split('→')
                    current_str = parts[0].strip()
                    target_part = parts[1].strip()
                    
                    # Remove percentage if present
                    target_str = target_part.split('(')[0].strip()
                    
                    try:
                        current_val = float(current_str)
                        target_val = float(target_str)
                        change_pct = ((target_val - current_val) / current_val * 100) if current_val != 0 else 0
                        
                        interventions.append({
                            'feature': feature,
                            'current': current_val,
                            'target': target_val,
                            'change_pct': change_pct,
                            'intervention': change.get('intervention_type', 'Lifestyle modification'),
                            'methods': change.get('methods', []),
                            'timeframe': change.get('timeframe', '3-6 months'),
                            'feasibility': best_cf.get('Feasibility', 'Moderate'),
                            'impact': change.get('impact', 'Moderate'),
                            'priority': f"Priority {best_cf.get('Priority', 2)}"
                        })
                    except (ValueError, TypeError):
                        # Skip if can't parse numbers
                        continue
            except Exception as e:
                # Skip malformed change entries
                continue
    
    return interventions


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_llm():
    """Test function to verify LLM is working"""
    print("\n" + "="*80)
    print("TESTING CLINICAL LLM")
    print("="*80)
    
    try:
        llm = ClinicalLLM()
        
        # Test data
        patient_data = "Age: 45, Gender: Female, Glucose: 12.5, BMI: 32.0"
        probability = 0.75
        lime_features = [
            ("Glucose > 10.0", 0.35),
            ("BMI > 30.0", 0.18),
            ("Age 40-50", 0.12)
        ]
        
        print("\nGenerating test narrative...")
        narrative = llm.generate_patient_narrative(
            patient_data, 
            probability, 
            lime_features
        )
        
        print("\n" + "-"*80)
        print("PATIENT NARRATIVE:")
        print("-"*80)
        print(narrative)
        print("-"*80)
        
        print("\n✅ LLM test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ LLM test failed: {e}")
        return False


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
========================================================================
Enhanced Clinical LLM with Counterfactual Integration
========================================================================

Usage in your pipeline:

# 1. Initialize LLM
clinical_llm = ClinicalLLM()

# 2. Get patient data and predictions
patient_row = X_test.iloc[i]
probability = rf_model.predict_proba(...)[0][1]
lime_features = lime_explainer.explain_instance(...).as_list()

# 3. Generate counterfactuals (from your DiCE analysis)
cf_analysis = [...]  # Your counterfactual analysis results

# 4. Extract interventions for LLM
interventions = extract_counterfactual_interventions(cf_analysis, patient_row)

# 5. Generate patient narrative
patient_narrative = clinical_llm.generate_patient_narrative(
    patient_data_str=patient_row.to_string(),
    probability=probability,
    lime_features=lime_features,
    counterfactual_interventions=interventions
)

# 6. Generate doctor report
doctor_report = clinical_llm.generate_doctor_report(
    patient_data_str=patient_row.to_string(),
    probability=probability,
    lime_features=lime_features,
    counterfactual_interventions=interventions
)
========================================================================
    """)
    
    # Run test
    test_llm()