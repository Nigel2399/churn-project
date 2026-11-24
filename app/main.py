"""The API.  Run:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict

MAX_BATCH = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # fail at startup (not on the first request) if the model is missing
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


class Customer(BaseModel):
    age: Optional[int] = Field(None, ge=18, le=100)
    tenure_months: Optional[int] = Field(None, ge=0, le=120)
    monthly_charges: Optional[float] = Field(None, gt=0)
    total_charges: Optional[float] = Field(None, ge=0)
    support_calls: Optional[int] = Field(None, ge=0, le=50)
    gender: Optional[Literal["Male", "Female"]] = None
    contract: Optional[Literal["Month-to-month", "One year", "Two year"]] = None
    payment_method: Optional[
        Literal["Credit card", "Bank transfer", "Electronic check", "Mailed check"]
    ] = None
    internet_service: Optional[Literal["DSL", "Fiber", "No internet"]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 30,
                "tenure_months": 2,
                "monthly_charges": 85.0,
                "total_charges": 170.0,
                "support_calls": 4,
                "gender": "Male",
                "contract": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber",
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_one(customer: Customer):
    return predict([customer.model_dump()])[0]


@app.post("/predict/batch")
def predict_many(customers: list[Customer]):
    if not customers:
        raise HTTPException(status_code=422, detail="Send at least one customer.")
    if len(customers) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Max {MAX_BATCH} customers per request.")
    return predict([c.model_dump() for c in customers])