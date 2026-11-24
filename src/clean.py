"""Step 4 - turn the messy CSV into a clean one.  Run:  python -m src.clean

Note: missing values are NOT filled here. Filling happens inside the model
pipeline (src/train.py) so the test data never leaks into the fill values.
"""
import numpy as np
import pandas as pd

from src.config import CLEAN_PATH, RAW_PATH, TARGET

GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
PAYMENT_MAP = {
    "credit card": "Credit card",
    "bank transfer": "Bank transfer",
    "electronic check": "Electronic check",
    "e-check": "Electronic check",
    "mailed check": "Mailed check",
}
INTERNET_MAP = {
    "dsl": "DSL",
    "fiber": "Fiber",
    "fiber optic": "Fiber",
    "no internet": "No internet",
}
TARGET_MAP = {"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0}

# Values outside these ranges are treated as errors and become missing.
RANGES = {
    "age": (18, 100),
    "tenure_months": (0, 120),
    "monthly_charges": (1, 1000),
    "total_charges": (0, 100_000),
    "support_calls": (0, 50),
}


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    return df


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def to_number(s: pd.Series) -> pd.Series:
    """'$1,200.50' -> 1200.5 ; ' ' -> NaN ; 'abc' -> NaN"""
    text = s.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def parse_dates(s: pd.Series) -> pd.Series:
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"]
    out = pd.to_datetime(s, format=formats[0], errors="coerce")
    for fmt in formats[1:]:
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out


def clean_contract(s: pd.Series) -> pd.Series:
    t = _norm(s).str.replace(r"[-_\s]+", " ", regex=True)
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[t.str.contains("month", na=False)] = "Month-to-month"
    out[t.str.contains(r"^(?:one|1)\b", na=False)] = "One year"
    out[t.str.contains(r"^(?:two|2)\b", na=False)] = "Two year"
    return out


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = standardise_columns(raw)

    # 1. Duplicates (exact copies first, then repeated customer ids)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="customer_id", keep="first")

    out = pd.DataFrame(index=df.index)
    out["customer_id"] = df["customer_id"].astype(str).str.strip()
    out["signup_date"] = parse_dates(df["signup_date"])

    # 2. Numbers: text -> float, impossible values -> missing
    for col, (low, high) in RANGES.items():
        values = to_number(df[col])
        out[col] = values.where(values.between(low, high))

    # 3. Categories: one spelling per value
    out["gender"] = _norm(df["gender"]).map(GENDER_MAP)
    out["contract"] = clean_contract(df["contract"])
    out["payment_method"] = _norm(df["payment_method"]).map(PAYMENT_MAP)
    out["internet_service"] = _norm(df["internet_service"]).map(INTERNET_MAP)

    # 4. Target: 0/1, rows without a label are useless for training
    out[TARGET] = _norm(df[TARGET]).map(TARGET_MAP)
    out = out.dropna(subset=[TARGET])
    out[TARGET] = out[TARGET].astype(int)

    return out.reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(RAW_PATH, dtype=str)  # read everything as text, convert on purpose
    clean = clean_dataframe(raw)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)

    print(f"Rows: {len(raw)} raw -> {len(clean)} clean")
    print("\nMissing values after cleaning:")
    print(clean.isna().sum().to_string())
    print(f"\nChurn rate: {clean[TARGET].mean():.1%}")
    print(f"Saved -> {CLEAN_PATH}")


if __name__ == "__main__":
    main()