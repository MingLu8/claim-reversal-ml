from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import os

cwd = os.getcwd()
print(cwd)

modelFile = "../ai/models/claim_reversal_model.joblib"
if os.path.exists(modelFile):
    print("model file found.")
else:
    print("model file not found.")
router = APIRouter()

model = joblib.load(modelFile)

THRESHOLD = 0.7
@router.post("/predict-reversal")
def predict(data: dict):
    FEATURE_COLUMNS = [
        "ingredient_cost",
        "copay_amount",
        "days_supply",
        "member_tenure_days",
        "refill_number",
        "prior_reversal_count",
        "pharmacy_type",
        "drug_tier",
        "pharmacy_historical_reversal_rate",
    ]

    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    print(f"Expected features are: {model.named_steps['preprocessor'].feature_names_in_}")
    try:
        prob = model.predict_proba(df)[0][1]
    except Exception as e:
        print(f"Error occurred while making prediction: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while making prediction:" + str(e))

    decision = "REVIEW" if prob >= THRESHOLD else "ALLOW"

    return {
        "reversal_probability": float(prob),
        "model_version": "v1.0",
        "decision": decision
    }