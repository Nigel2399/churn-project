"""Run a pre-trained Hugging Face model on customer messages.

Run:  python scripts/sentiment_demo.py
"""
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

messages = [
    "I love the new fiber plan, the speed is amazing!",
    "My internet has been down for three days and nobody is helping me.",
    "Billing is confusing, I want to cancel my contract.",
    "Support fixed my issue quickly, thank you.",
    "I called four times and the problem is still not solved.",
]

for text, result in zip(messages, classifier(messages)):
    print(f"{result['label']:<9} {result['score']:.3f}  {text}")