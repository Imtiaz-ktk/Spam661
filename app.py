import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load saved model
model_path = "email-spam-detector"

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Streamlit UI
st.title("Spam Email Detector")

message = st.text_area("Enter Email or SMS")

if st.button("Check"):

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits).item()

    if prediction == 1:
        st.error("Spam Detected")
    else:
        st.success("Not Spam")
