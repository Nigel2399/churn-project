"""LangChain chain: churn model -> prompt -> local LLM -> retention suggestion.

Run from the project root:  python -m scripts.retention_chain
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline

from src.predict import predict

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 80, "return_full_text": False, "do_sample": False},
)

prompt = PromptTemplate.from_template(
    "You are a customer retention assistant for a telecom company.\n"
    "A customer has a {risk} churn risk ({percent}). "
    "Contract: {contract}. Tenure: {tenure_months} months. "
    "Support calls: {support_calls}. Monthly charges: {monthly_charges}.\n"
    "Write one short, friendly retention action in at most 2 sentences.\n"
    "Action:"
)


def add_risk(customer: dict) -> dict:
    p = predict([customer])[0]["churn_probability"]
    risk = "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"
    return {**customer, "percent": f"{p:.0%}", "risk": risk}


chain = RunnableLambda(add_risk) | prompt | llm | StrOutputParser()

customers = [
    {"age": 30, "tenure_months": 2, "monthly_charges": 85.0, "total_charges": 170.0,
     "support_calls": 4, "gender": "Male", "contract": "Month-to-month",
     "payment_method": "Electronic check", "internet_service": "Fiber"},
    {"age": 60, "tenure_months": 65, "monthly_charges": 45.0, "total_charges": 2900.0,
     "support_calls": 0, "gender": "Female", "contract": "Two year",
     "payment_method": "Credit card", "internet_service": "DSL"},
]

for customer in customers:
    print("-" * 60)
    print("Contract:", customer["contract"], "| Tenure:", customer["tenure_months"], "months")
    print("Suggestion:", chain.invoke(customer).strip())