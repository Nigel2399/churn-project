"""Central settings. If you swap in your own dataset, change things here first."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "raw" / "customers_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "customers_clean.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

TARGET = "churned"
NUMERIC = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
CATEGORICAL = ["gender", "contract", "payment_method", "internet_service"]

CATEGORY_VALUES = {
    "gender": ["Male", "Female"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "payment_method": ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
    "internet_service": ["DSL", "Fiber", "No internet"],
}

RANDOM_STATE = 42
THRESHOLD = 0.4  # probability above which we predict "Yes" (churn)