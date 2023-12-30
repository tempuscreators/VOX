import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare test input
test_text = "Hello, world! This is a test for the Mistral model."
inputs = tokenizer(test_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Check the output
print(outputs.logits)
