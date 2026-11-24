"""Step 2 - generate a MESSY raw dataset.  Run:  python -m src.make_data"""
import numpy as np
import pandas as pd

from src.config import CATEGORY_VALUES, RANDOM_STATE, RAW_PATH

N_ROWS = 3000


def generate_clean(n: int = N_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(CATEGORY_VALUES["contract"], size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(CATEGORY_VALUES["internet_service"], size=n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(CATEGORY_VALUES["payment_method"], size=n, p=[0.30, 0.25, 0.30, 0.15])
    gender = rng.choice(CATEGORY_VALUES["gender"], size=n)

    age = rng.integers(18, 80, size=n)
    tenure = rng.integers(1, 73, size=n)
    base = np.select([internet == "Fiber", internet == "DSL"], [70, 50], default=25)
    monthly = np.clip(np.round(base + rng.normal(0, 8, size=n), 2), 15, None)
    total = np.clip(np.round(monthly * tenure + rng.normal(0, 20, size=n), 2), 0, None)
    calls = rng.poisson(1.5, size=n)

    logit = (
        -0.8
        + 1.2 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        + 0.5 * (internet == "Fiber")
        + 0.4 * (payment == "Electronic check")
        - 0.03 * tenure
        + 0.015 * (monthly - 50)
        + 0.35 * calls
        - 0.01 * (age - 40)
    )
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    signup = pd.Timestamp("2026-01-01") - pd.to_timedelta(tenure * 30, unit="D")

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup_date": signup,
            "age": age,
            "gender": gender,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": calls,
            "churned": churned,
        }
    )


def make_messy(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    df = df.copy()

    def corrupt(values, fn, p):
        return [fn(v) if rng.random() < p else v for v in values]

    def blank(values, p):
        return [np.nan if rng.random() < p else v for v in values]

    variants = {
        "gender": {"Male": ["male", "M", " MALE "], "Female": ["female", "F", "FEMALE "]},
        "contract": {
            "Month-to-month": ["month to month", "MONTH-TO-MONTH"],
            "One year": ["1 year", "one year "],
            "Two year": ["2 year", "two year "],
        },
        "payment_method": {
            "Credit card": ["credit card", "CREDIT CARD "],
            "Bank transfer": ["bank transfer", " Bank Transfer"],
            "Electronic check": ["electronic check", "e-check"],
            "Mailed check": ["mailed check", "MAILED CHECK "],
        },
        "internet_service": {
            "DSL": ["dsl", "DSL "],
            "Fiber": ["fiber", "FIBER", "fiber optic"],
            "No internet": ["no internet", "NO INTERNET "],
        },
    }
    for col, mapping in variants.items():
        values = corrupt(df[col], lambda v, m=mapping: rng.choice(m[v]), 0.30)
        df[col] = blank(values, 0.03)

    df["monthly_charges"] = blank(corrupt(df["monthly_charges"], lambda v: f"${v:,.2f}", 0.15), 0.03)
    df["total_charges"] = blank(corrupt(df["total_charges"], lambda v: " ", 0.01), 0.02)
    df["age"] = blank(corrupt(df["age"], lambda v: rng.choice([-1, 999]), 0.01), 0.04)
    df["tenure_months"] = blank(corrupt(df["tenure_months"], lambda v: -5, 0.005), 0.02)
    df["support_calls"] = blank(df["support_calls"], 0.03)

    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"]
    df["signup_date"] = blank([d.strftime(fmts[rng.integers(0, 3)]) for d in df["signup_date"]], 0.02)

    yes_no = {1: ["Yes", "yes", "Y", "1"], 0: ["No", "no", "N", "0"]}
    df["churned"] = blank([rng.choice(yes_no[v]) for v in df["churned"]], 0.01)

    df.columns = [
        "Customer ID", "Signup Date", "Age", "Gender", "Contract", "Payment Method",
        "Internet Service", "Tenure Months", "Monthly Charges", "Total Charges",
        "Support Calls", "Churned",
    ]

    dups = df.sample(frac=0.03, random_state=seed)
    df = pd.concat([df, dups]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    raw = make_messy(generate_clean())
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(raw)} messy rows -> {RAW_PATH}")


if __name__ == "__main__":
    main()