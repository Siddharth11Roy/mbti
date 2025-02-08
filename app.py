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


# def main():
#     st.set_page_config(page_title="MBTI Personality Predictor", layout="centered")
#     st.title("🧠 MBTI Personality Predictor")

#     # **Load Model**
#     classifier = MBTIClassifier()

#     # **User Input**
#     user_input = st.text_area("Enter your text:", placeholder="Type something...")

#     # **Predict Button**
#     if st.button("Predict Personality Type"):
#         if user_input.strip():
#             with st.spinner("Analyzing..."):
#                 predicted_mbti = classifier.predict_mbti(user_input)
                
#                 # **Display Result**
#                 if predicted_mbti is not None:
#                     st.success(f"📝 Predicted MBTI Type: **{predicted_mbti}**")
#                 else:
#                     st.error("⚠️ Error: Could not classify text.")
#         else:
#             st.warning("⚠️ Please enter some text.")

# # **Run the App**
# if __name__ == "__main__":
#     main()

questions = {
    "Personal Reflections & Identity": [
        "How do you usually describe yourself to others?",
        "Do you consider yourself more introverted or extroverted? Why?",
        "How do you handle criticism or feedback from others?",
        "What is something about yourself that you feel most people don’t understand?",
        "How do you feel about your relationship with your family and close friends?"
    ],
    "Life Challenges & Decision-Making": [
        "What has been the hardest decision you’ve had to make in life?",
        "Have you ever struggled with deciding your career path? How did you deal with it?",
        "If you could go back in time and change one decision, what would it be and why?",
        "How do you react when faced with uncertainty or major life changes?"
    ],
    "Emotions & Thought Processes": [
        "What usually makes you feel overwhelmed or stressed?",
        "How do you usually deal with sadness or disappointment?",
        "Do you find comfort in being alone, or do you prefer being around people when you're upset?",
        "What do you think is your greatest strength and your biggest weakness?"
    ],
    "Interests & Preferences": [
        "What books, movies, or TV shows have had a deep impact on you?",
        "Do you have a favorite song that reflects your current mood or personality?",
        "If you had to pick a fictional character that closely matches your personality, who would it be and why?",
        "How do you usually spend your free time?"
    ],
    "Beliefs & Philosophies": [
        "Do you believe people are generally good or bad? Why?",
        "What are your thoughts on personal freedom versus societal expectations?",
        "How do you define happiness in your life?",
        "Do you believe that everything happens for a reason? Why or why not?"
    ]
}

# Store responses in session state
if 'responses' not in st.session_state:
    st.session_state.responses = {}

def save_response(question, response):
    if len(response.split()) < 15:
        st.warning("Elongate the answer")
    elif response == "0":
        st.session_state.responses[question] = "Skipped"
    else:
        st.session_state.responses[question] = response
        st.success("Response saved")

# Streamlit UI
st.title("Personality Type Assessment")
tabs = st.tabs(list(questions.keys()))

for idx, (category, qs) in enumerate(questions.items()):
    with tabs[idx]:
        for q in qs:
            response = st.text_area(q, key=q)
            if st.button(f"Save: {q}", key=f"btn_{q}"):
                save_response(q, response)

if st.button("Submit Answers"):
    if not st.session_state.responses:
        st.warning("Please answer at least one question before submitting.")
    else:
        combined_text = " ".join(f"{q}: {a}" for q, a in st.session_state.responses.items())
        preprocessed_text = preprocess_text(combined_text)
        
        # Tokenize & Predict
        inputs = tokenizer(preprocessed_text, return_tensors='pt', padding='max_length', truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        
        st.write(f"### Predicted Personality Type: {prediction}")

