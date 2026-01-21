import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Microsoft Phi-3 Mini (3.8B Parameters)
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

class ClinicalLLM:
    def __init__(self):
        print(f"Loading {MODEL_ID} to RTX 4080...")
        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, 
                trust_remote_code=True
            )
            
            # 2. Load Model 
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16, 
                device_map="cuda", 
                trust_remote_code=True,
                attn_implementation="eager" 
            )
            
            # 3. Create Pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=400,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.system_prompt = """You are a Clinical Decision Support System (CDSS) Assistant. 
            Interpret the diabetes risk prediction based STRICTLY on the provided data. 
            Do not diagnose or prescribe. Use professional medical language."""
            
            print("SUCCESS: Local LLM loaded on GPU!")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading LLM: {e}")
            self.model = None

    def generate_explanation(self, patient_data_str, probability, lime_features):
        if not self.model:
            return "Error: LLM model not loaded."

        # Format inputs
        risk_str = f"{probability*100:.1f}%"
        risk_level = "HIGH" if probability > 0.5 else "LOW"
        features_text = "\n".join([f"- {feat}: {weight:.4f}" for feat, weight in lime_features])

        # Create Prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            **Patient Data:**
            {patient_data_str}

            **Risk Assessment:**
            - Prediction: {risk_level} ({risk_str})
            
            **Key Drivers (LIME Analysis):**
            {features_text}

            Write a short clinical summary for a doctor explaining why this patient is at risk.
            """}
        ]

        # Generate (CRITICAL FIX: use_cache=False disables the broken component)
        outputs = self.pipe(messages, use_cache=False)
        return outputs[0]["generated_text"][-1]["content"]