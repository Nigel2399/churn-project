"""API tests (need the trained model, plus fastapi + httpx)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

CUSTOMER = {
    "age": 30, "tenure_months": 2, "monthly_charges": 85.0, "total_charges": 170.0,
    "support_calls": 4, "gender": "Male", "contract": "Month-to-month",
    "payment_method": "Electronic check", "internet_service": "Fiber",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" makes startup (model loading) run
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_one(client):
    response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in {"Yes", "No"}


def test_predict_with_missing_fields(client):
    assert client.post("/predict", json={"age": 40}).status_code == 200


def test_rejects_invalid_age(client):
    assert client.post("/predict", json={**CUSTOMER, "age": 5}).status_code == 422


def test_rejects_invalid_category(client):
    assert client.post("/predict", json={**CUSTOMER, "contract": "Five year"}).status_code == 422


def test_rejects_wrong_type(client):
    assert client.post("/predict", json={**CUSTOMER, "age": "old"}).status_code == 422


def test_batch(client):
    response = client.post("/predict/batch", json=[CUSTOMER, {**CUSTOMER, "tenure_months": 60}])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json=[]).status_code == 422