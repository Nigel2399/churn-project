"""Prediction logic shared by the API and the tests."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import CATEGORICAL, MODEL_PATH, NUMERIC, THRESHOLD


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def to_frame(records: list[dict]) -> pd.DataFrame:
    """List of dicts -> DataFrame with exactly the columns the model expects."""
    df = pd.DataFrame(records).reindex(columns=NUMERIC + CATEGORICAL)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)  # None -> NaN
    return df


def predict(records: list[dict]) -> list[dict]:
    model = load_model()
    proba = model.predict_proba(to_frame(records))[:, 1]
    return [
        {
            "churn_probability": round(float(p), 4),
            "prediction": "Yes" if p >= THRESHOLD else "No",
        }
        for p in proba
    ]