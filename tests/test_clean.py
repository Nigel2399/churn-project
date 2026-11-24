"""Unit tests for the cleaning step (no files needed)."""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean_dataframe

COLUMNS = [
    "Customer ID", "Signup Date", "Age", "Gender", "Contract", "Payment Method",
    "Internet Service", "Tenure Months", "Monthly Charges", "Total Charges",
    "Support Calls", "Churned",
]

R0 = ["C1", "2025-03-01", "34", " MALE ", "month to month", "Credit Card ", "fiber optic", "12", "$70.50", "$1,200.00", "2", "Yes"]
R1 = ["C2", "01/04/2025", "-1", "f", "1 year", "e-check", "dsl", "3", "50", " ", "0", "no"]
R2 = list(R1)  # exact duplicate of R1
R3 = ["C3", "20250110", "40", "male", "2 year", "mailed check", "no internet", "8", "30", "90", "1", np.nan]  # no label
R4 = ["C4", "2025-05-05", "999", "Female", "two year ", "bank transfer", "DSL", "-5", "$45.00", "300", "4", "Y"]


@pytest.fixture()
def cleaned():
    raw = pd.DataFrame([R0, R1, R2, R3, R4], columns=COLUMNS)
    return clean_dataframe(raw)


def test_duplicates_and_unlabelled_rows_removed(cleaned):
    assert cleaned["customer_id"].tolist() == ["C1", "C2", "C4"]


def test_column_names_are_snake_case(cleaned):
    assert "monthly_charges" in cleaned.columns
    assert "Monthly Charges" not in cleaned.columns


def test_categories_are_standardised(cleaned):
    assert cleaned["gender"].tolist() == ["Male", "Female", "Female"]
    assert cleaned["contract"].tolist() == ["Month-to-month", "One year", "Two year"]
    assert cleaned["payment_method"].tolist() == ["Credit card", "Electronic check", "Bank transfer"]
    assert cleaned["internet_service"].tolist() == ["Fiber", "DSL", "DSL"]


def test_money_text_becomes_numbers(cleaned):
    assert cleaned["monthly_charges"].tolist() == [70.5, 50.0, 45.0]
    assert cleaned.loc[0, "total_charges"] == 1200.0
    assert pd.isna(cleaned.loc[1, "total_charges"])  # blank string -> missing


def test_impossible_values_become_missing(cleaned):
    assert cleaned.loc[0, "age"] == 34
    assert pd.isna(cleaned.loc[1, "age"])  # -1
    assert pd.isna(cleaned.loc[2, "age"])  # 999
    assert pd.isna(cleaned.loc[2, "tenure_months"])  # -5


def test_target_is_zero_one(cleaned):
    assert cleaned["churned"].tolist() == [1, 0, 1]


def test_dates_parsed_from_mixed_formats(cleaned):
    assert cleaned["signup_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-03-01", "2025-04-01", "2025-05-05",
    ]