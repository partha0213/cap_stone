"""
part4_llm.py
============
Part 4 — LLM-Powered Feature: Model Prediction Explanation Pipeline (Track C)
Capstone Project · Housing Price Dataset

Outputs:
  • Console execution logs showing the PII Guardrail, temperature A/B test, 
    and structured validation for 3 distinct test records.
"""

import os
import re
import json
import requests
import joblib
import pandas as pd
import numpy as np
from jsonschema import validate, ValidationError

# ==============================================================================
# SCHEMA DEFINITION
# ==============================================================================
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string"},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}

FALLBACK_JSON = {
    "prediction_label": "null",
    "confidence_level": "null",
    "top_reason": "null",
    "second_reason": "null",
    "next_step": "null"
}

# ==============================================================================
# PII GUARDRAIL
# ==============================================================================
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

# ==============================================================================
# CALL LLM FUNCTION (WITH MOCK FALLBACK IN CASE KEY IS MISSING/EXPIRED)
# ==============================================================================
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    api_key = os.environ.get("LLM_API_KEY")
    
    # Check guardrail
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None
        
    if not api_key:
        # Secure, descriptive Mock response if no API key is configured
        print("  [API Key missing] Using local mock LLM response...")
        return get_mock_response(user_prompt)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3-8b-instruct:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"  [API Error, status code {response.status_code}] Falling back to mock...")
            return get_mock_response(user_prompt)
    except Exception as e:
        print(f"  [HTTP Exception: {e}] Falling back to mock...")
        return get_mock_response(user_prompt)

def get_mock_response(user_prompt):
    # Parse info from user prompt to generate a realistic structured JSON
    # extract class
    cls_match = re.search(r"Predicted Class:\s*(\d)", user_prompt)
    prob_match = re.search(r"Predicted Probability:\s*([\d\.]+)", user_prompt)
    
    pred_cls = cls_match.group(1) if cls_match else "1"
    prob_val = float(prob_match.group(1)) if prob_match else 0.85
    
    label_str = "High Value Home" if pred_cls == "1" else "Low/Average Value Home"
    conf_level = "high" if prob_val > 0.8 or prob_val < 0.2 else "medium"
    
    mock_data = {
        "prediction_label": label_str,
        "confidence_level": conf_level,
        "top_reason": "OverallQual and GrLivArea are the dominant drivers in this prediction.",
        "second_reason": "Neighborhood has a moderate but secondary coefficient impact.",
        "next_step": "Validate house features manually or collect more premium property coordinates."
    }
    return json.dumps(mock_data, indent=2)

# ==============================================================================
# PIPELINE DEMO
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PART 4 — LLM-POWERED FEATURE (TRACK C: MODEL PREDICTION EXPLANATION)")
    print("="*80)
    
    # 1. Verify LLM setup with simple test prompt
    print("\n[LLM Verification Test]")
    test_res = call_llm("You are a helpful assistant.", "Reply with only the word: hello", temperature=0)
    print(f"Response: {test_res.strip() if test_res else 'None'}")
    
    # 2. Verify PII Guardrail
    print("\n[PII Guardrail Testing]")
    pii_input = "Tell me about this record for user ashwin@example.com."
    clean_input = "Tell me about this record for a house in Gilbert."
    
    print("Testing PII email input:")
    res_pii = call_llm("System", pii_input)
    
    print("Testing clean input:")
    res_clean = call_llm("System", clean_input)
    print(f"Clean response: {res_clean.strip() if res_clean else 'None'}")
    
    # 3. Load Model and clean data to select test rows
    print("\n[Loading best_model.pkl and testing on 3 records]")
    pipeline = joblib.load("best_model.pkl")
    
    df = pd.read_csv("cleaned_data.csv")
    X = df.drop(columns=["SalePrice", "Id", "Alley"], errors="ignore")
    
    # Preprocess categorical features via One-Hot encoding exactly as done at fit time
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    
    # Select 3 distinct inputs from dataset
    # We choose 3 index locations: 10, 100, 500
    indices = [10, 100, 500]
    test_records = X_encoded.iloc[indices].copy()
    test_records_raw = X.iloc[indices].copy()
    
    # Define prompt templates
    sys_prompt = (
        "You are an AI model explanation system. "
        "Explain the model's prediction using the provided features. "
        "You must respond ONLY with a single valid JSON object matching this schema:\n"
        "{\n"
        "  \"prediction_label\": \"string (e.g. High Value Home or Low/Average Value Home)\",\n"
        "  \"confidence_level\": \"string (high, medium, or low)\",\n"
        "  \"top_reason\": \"string (key driver based on overall quality and size)\",\n"
        "  \"second_reason\": \"string (secondary feature driver)\",\n"
        "  \"next_step\": \"string (actionable business next step)\"\n"
        "}\n"
        "Do not include any extra text, markdown formatting, or HTML tags outside the JSON block."
    )
    
    results = []
    
    for idx, (loc, row) in enumerate(zip(indices, test_records.iterrows())):
        row_df = pd.DataFrame([row[1]])
        pred = pipeline.predict(row_df)[0]
        prob = pipeline.predict_proba(row_df)[0][1]
        
        # Format the user prompt using the clean, raw values (not one-hot encoded)
        raw_row = test_records_raw.iloc[idx].to_dict()
        user_prompt = (
            f"Record Features:\n{json.dumps(raw_row, indent=2)}\n\n"
            f"Predicted Class: {pred}\n"
            f"Predicted Probability: {prob:.4f}"
        )
        
        print(f"\n--- RECORD {idx+1} (Index {loc}) ---")
        print(f"Predicted Class: {pred}, Probability: {prob:.4f}")
        
        # Temp = 0.0 run
        resp_t0 = call_llm(sys_prompt, user_prompt, temperature=0.0)
        
        # Validate output
        try:
            parsed = json.loads(resp_t0.strip())
            validate(instance=parsed, schema=OUTPUT_SCHEMA)
            status = "PASS"
        except (json.JSONDecodeError, ValidationError) as e:
            status = f"FAIL: {type(e).__name__}"
            parsed = FALLBACK_JSON
            
        print(f"Validated JSON (T=0.0) Status: {status}")
        print(json.dumps(parsed, indent=2))
        
        # Temp = 0.7 run
        resp_t7 = call_llm(sys_prompt, user_prompt, temperature=0.7)
        
        results.append({
            "idx": idx+1,
            "pred": pred,
            "prob": prob,
            "t0_resp": resp_t0,
            "t7_resp": resp_t7,
            "status": status
        })
        
    # Temperature comparison display
    print("\n" + "="*80)
    print("TEMPERATURE A/B COMPARISON TABLE DATA")
    print("="*80)
    for res in results:
        print(f"\nRecord {res['idx']}: Pred={res['pred']}, Prob={res['prob']:.4f}")
        print(f"  T=0.0 Output: {res['t0_resp'].strip()[:200]}...")
        print(f"  T=0.7 Output: {res['t7_resp'].strip()[:200]}...")
        
    print("\n✓ LLM Prediction Explanation Pipeline complete.")
