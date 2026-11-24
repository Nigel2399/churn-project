"""Hit a RUNNING API (local or Docker) like a real client would.

Start the API first, then:  python scripts/smoke_test.py
Other URL:                  python scripts/smoke_test.py http://localhost:8000
"""
import json
import sys
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

CUSTOMER = {
    "age": 30, "tenure_months": 2, "monthly_charges": 85.0, "total_charges": 170.0,
    "support_calls": 4, "gender": "Male", "contract": "Month-to-month",
    "payment_method": "Electronic check", "internet_service": "Fiber",
}


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    health = call("/health")
    assert health == {"status": "ok"}, health
    print("health  OK ->", health)

    result = call("/predict", CUSTOMER)
    assert 0 <= result["churn_probability"] <= 1, result
    print("predict OK ->", result)

    batch = call("/predict/batch", [CUSTOMER, {**CUSTOMER, "tenure_months": 60, "contract": "Two year"}])
    assert len(batch) == 2, batch
    print("batch   OK ->", batch)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()