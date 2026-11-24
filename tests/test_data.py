"""Data validation: checks on the cleaned CSV produced by `python -m src.clean`."""
import pandas as pd
import pytest

from src.clean import RANGES
from src.config import CATEGORY_VALUES, CLEAN_PATH, TARGET


@pytest.fixture(scope="module")
def df():
    if not CLEAN_PATH.exists():
        pytest.fail(f"{CLEAN_PATH} not found. Run: python -m src.make_data && python -m src.clean")
    return pd.read_csv(CLEAN_PATH)


def test_enough_rows(df):
    assert len(df) > 1000


def test_no_duplicate_customers(df):
    assert df["customer_id"].is_unique


def test_target_has_no_missing_and_only_0_1(df):
    assert df[TARGET].notna().all()
    assert set(df[TARGET].unique()) <= {0, 1}


def test_both_classes_present(df):
    assert 0.05 < df[TARGET].mean() < 0.95


@pytest.mark.parametrize("col", list(RANGES))
def test_numeric_columns_in_range(df, col):
    low, high = RANGES[col]
    assert df[col].dropna().between(low, high).all()


@pytest.mark.parametrize("col", list(CATEGORY_VALUES))
def test_categories_only_use_allowed_values(df, col):
    assert set(df[col].dropna().unique()) <= set(CATEGORY_VALUES[col])