"""Download the model from the Hub and predict.

Run:  python scripts/load_from_hf.py YOUR-HF-USERNAME/churn-model
"""
import sys

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id=sys.argv[1], filename="model.joblib")
model = joblib.load(path)

customer = pd.DataFrame([{
    "age": 30, "tenure_months": 2, "monthly_charges": 85.0, "total_charges": 170.0,
    "support_calls": 4, "gender": "Male", "contract": "Month-to-month",
    "payment_method": "Electronic check", "internet_service": "Fiber",
}])
print("Churn probability:", round(float(model.predict_proba(customer)[0, 1]), 4))