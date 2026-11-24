"""Tests for the trained model (run `python -m src.train` first)."""
import json

import pytest

from src.config import METRICS_PATH, MODEL_PATH
from src.predict import predict

RISKY = {
    "age": 25, "tenure_months": 2, "monthly_charges": 90.0, "total_charges": 180.0,
    "support_calls": 5, "gender": "Male", "contract": "Month-to-month",
    "payment_method": "Electronic check", "internet_service": "Fiber",
}
LOYAL = {
    "age": 60, "tenure_months": 65, "monthly_charges": 45.0, "total_charges": 2900.0,
    "support_calls": 0, "gender": "Female", "contract": "Two year",
    "payment_method": "Credit card", "internet_service": "DSL",
}


@pytest.fixture(scope="module", autouse=True)
def require_model():
    if not MODEL_PATH.exists():
        pytest.fail("Model not found. Run: python -m src.train")


def test_minimum_quality():
    metrics = json.loads(METRICS_PATH.read_text())
    assert metrics["test"]["roc_auc"] >= 0.70


def test_output_format():
    result = predict([RISKY, LOYAL])
    assert len(result) == 2
    for row in result:
        assert 0.0 <= row["churn_probability"] <= 1.0
        assert row["prediction"] in {"Yes", "No"}


def test_risky_customer_scores_higher_than_loyal_one():
    risky, loyal = predict([RISKY, LOYAL])
    assert risky["churn_probability"] > loyal["churn_probability"]


def test_handles_missing_values():
    result = predict([{"age": None, "contract": None, "monthly_charges": 60.0}])
    assert 0.0 <= result[0]["churn_probability"] <= 1.0


def test_handles_completely_empty_record():
    assert len(predict([{}])) == 1


def test_handles_unseen_category():
    result = predict([{**RISKY, "contract": "Five year"}])
    assert 0.0 <= result[0]["churn_probability"] <= 1.0