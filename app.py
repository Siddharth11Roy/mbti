import streamlit as st
import pandas as pd
import pickle
import torch
import plotly.graph_objects as go
import re
import time
import logging
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
from datasets import Dataset
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
    text = text.replace("|||", " ")  # Replace separators
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caching function for inference
@st.cache_data(show_spinner=False)
def cached_predict_mbti(text, _model, _tokenizer, _device):
    try:
        # Input validation
        if not text or len(text.strip()) < 10:
            raise ValueError("Input text is too short")

        # Preprocess
        text = preprocess_text

        # Tokenize
        inputs = _tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(_device)

        # Predict
        start_time = time.time()
        with torch.no_grad():
            outputs = _model(**inputs)
        prediction_time = time.time() - start_time
        logger.info(f"Prediction completed in {prediction_time:.2f} seconds")

        # Get predicted class
        predictions = torch.argmax(outputs.logits, dim=1).item()
        return predictions  # This should be mapped to MBTI types

    except Exception as e:
        logger.error(f"Error during MBTI classification: {str(e)}")
        raise

class MBTIClassifier:
    def __init__(self, model_name="Sid26Roy/mbti"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Load model and tokenizer from Hugging Face Hub
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)

            self.model.to(self.device)
            self.model.eval()
            logger.info("MBTI model loaded successfully from Hugging Face Hub")

        except Exception as e:
            logger.error(f"Error loading MBTI model: {str(e)}")
            raise

    def predict_mbti(self, text):
        return cached_predict_mbti(text, self.model, self.tokenizer, self.device)


def main():
    st.set_page_config(page_title="MBTI Personality Predictor", layout="centered")
    st.title("🧠 MBTI Personality Predictor")

    # **Load Model**
    classifier = MBTIClassifier()

    # **User Input**
    user_input = st.text_area("Enter your text:", placeholder="Type something...")

    # **Predict Button**
    if st.button("Predict Personality Type"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                predicted_mbti = classifier.predict_mbti(user_input)
                
                # **Display Result**
                if predicted_mbti is not None:
                    st.success(f"📝 Predicted MBTI Type: **{predicted_mbti}**")
                else:
                    st.error("⚠️ Error: Could not classify text.")
        else:
            st.warning("⚠️ Please enter some text.")

# **Run the App**
if __name__ == "__main__":
    main()
