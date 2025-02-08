# import streamlit as st
# import pandas as pd
# import pickle
# import torch
# import plotly.graph_objects as go
# import re
# import time
# import logging
# import streamlit as st
# from streamlit_lottie import st_lottie
# import requests
# from transformers import (
#     BertTokenizer,
#     BertForSequenceClassification,
# )
# from datasets import Dataset
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import hashlib

# nltk.download('punkt_tab')


# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

# def generate_unique_key(question, index):
#     question_hash = hashlib.md5(question.encode()).hexdigest()[:10]  # Shortened hash
#     return f"q_{index}_{question_hash}"

# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
#     text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
#     text = text.replace("|||", " ")  # Replace separators
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return " ".join(tokens)

# # Initialize NLP tools
# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Caching function for inference
# @st.cache_data(show_spinner=False)
# def cached_predict_mbti(text, _model, _tokenizer, _device):
#     try:
#         # Input validation
#         if not text or len(text.strip()) < 10:
#             raise ValueError("Input text is too short")

#         # Preprocess
#         text = preprocess_text(text)

#         # Tokenize
#         inputs = _tokenizer(
#             text,
#             max_length=512,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         ).to(_device)

#         # Predict
#         start_time = time.time()
#         with torch.no_grad():
#             outputs = _model(**inputs)
#         prediction_time = time.time() - start_time
#         logger.info(f"Prediction completed in {prediction_time:.2f} seconds")

#         # Get predicted class
#         predictions = torch.argmax(outputs.logits, dim=1).item()
#         return predictions  # This should be mapped to MBTI types

#     except Exception as e:
#         logger.error(f"Error during MBTI classification: {str(e)}")
#         raise

# class MBTIClassifier:
#     def __init__(self, model_name="Sid26Roy/mbti"):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         try:
#             # Load model and tokenizer from Hugging Face Hub
#             self.tokenizer = BertTokenizer.from_pretrained(model_name)
#             self.model = BertForSequenceClassification.from_pretrained(model_name)

#             self.model.to(self.device)
#             self.model.eval()
#             logger.info("MBTI model loaded successfully from Hugging Face Hub")

#         except Exception as e:
#             logger.error(f"Error loading MBTI model: {str(e)}")
#             raise

#     def predict_mbti(self, text):
#         return cached_predict_mbti(text, self.model, self.tokenizer, self.device)


# # def main():
# #     st.set_page_config(page_title="MBTI Personality Predictor", layout="centered")
# #     st.title("🧠 MBTI Personality Predictor")

# #     # **Load Model**
# #     classifier = MBTIClassifier()

# #     # **User Input**
# #     user_input = st.text_area("Enter your text:", placeholder="Type something...")

# #     # **Predict Button**
# #     if st.button("Predict Personality Type"):
# #         if user_input.strip():
# #             with st.spinner("Analyzing..."):
# #                 predicted_mbti = classifier.predict_mbti(user_input)
                
# #                 # **Display Result**
# #                 if predicted_mbti is not None:
# #                     st.success(f"📝 Predicted MBTI Type: **{predicted_mbti}**")
# #                 else:
# #                     st.error("⚠️ Error: Could not classify text.")
# #         else:
# #             st.warning("⚠️ Please enter some text.")

# # # **Run the App**
# # if __name__ == "__main__":
# #     main()

# questions = {
#     "Personal Reflections & Identity": [
#         "How do you usually describe yourself to others?",
#         "Do you consider yourself more introverted or extroverted? Why?",
#         "How do you handle criticism or feedback from others?",
#     ],
#     "Life Challenges & Decision-Making": [
#         "What has been the hardest decision you’ve had to make in life?",
#         "Have you ever struggled with deciding your career path? How did you deal with it?",
#         "If you could go back in time and change one decision, what would it be and why?",
#     ],
#     "Emotions & Thought Processes": [
#         "What usually makes you feel overwhelmed or stressed?",
#         "How do you usually deal with sadness or disappointment?",
#         "Do you find comfort in being alone, or do you prefer being around people when you're upset?",
#     ],
#     "Interests & Preferences": [
#         "What books, movies, or TV shows have had a deep impact on you?",
#         "Do you have a favorite song that reflects your current mood or personality?",
#         "If you had to pick a fictional character that closely matches your personality, who would it be and why?",
#     ],
#     "Beliefs & Philosophies": [
#         "Do you believe people are generally good or bad? Why?",
#         "What are your thoughts on personal freedom versus societal expectations?",
#         "How do you define happiness in your life?",
#     ],
# }

# # Temporary storage for responses
# if "saved_answers" not in st.session_state:
#     st.session_state.saved_answers = {}

# # Tabs for different sections
# tabs = list(questions.keys())
# selected_tab = st.sidebar.radio("Select a section:", tabs)

# # Display questions for the selected section
# st.header(selected_tab)
# for i, q in enumerate(questions[selected_tab]):
#     answer = st.text_area(q, key=generate_unique_key(q, i))

#     # Save button for each question
#     if st.button(f"Save Answer {i+1}", key=f"save_{selected_tab}_{i}"):
#         if len(answer.split()) < 15:
#             st.warning("Elongate the answer")
#         elif answer.strip() == "0":
#             st.session_state.saved_answers[q] = "Skipped"
#             st.success("Answer Skipped!")
#         else:
#             st.session_state.saved_answers[q] = preprocess_text(answer)
#             st.success("Answer Saved!")

# # Final Submit button
# if st.button("Submit All Answers"):
#     if len(st.session_state.saved_answers) == 0:
#         st.warning("No answers saved. Please respond to at least one question.")
#     else:
#         # Combine all saved answers into a single paragraph
#         final_text = "\n".join([f"{q} {a}" for q, a in st.session_state.saved_answers.items()])
#         final_text = preprocess_text(final_text)

#         # Tokenize and send to model
#         inputs = tokenizer(
#             final_text,
#             max_length=512,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         ).to(device)

#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         st.success("Answers submitted successfully! Processing with the model...")
#         st.write(final_text)  # Display processed text (optional)


import streamlit as st
import torch
import re
import time
import logging
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import hashlib
from streamlit_lottie import st_lottie
import requests

# Download required NLTK resources
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the MBTI Model (Avoiding duplicate instantiations)
@st.cache_resource
def load_mbti_model():
    model_name = "Sid26Roy/mbti"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device

# Load the model once and use it throughout the app
tokenizer, model, device = load_mbti_model()

def load_lottieurl(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep punctuation in questions
    text = text.replace("|||", " ")  # Replace separators
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Function to generate unique keys for questions
def generate_unique_key(question, index):
    question_hash = hashlib.md5(question.encode()).hexdigest()[:10]  # Shortened hash
    return f"q_{index}_{question_hash}"

MBTI_CLASSES = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

lottie_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zdtukd5q.json")

# Initialize session state for saved answers
if "saved_answers" not in st.session_state:
    st.session_state.saved_answers = {}



# Streamlit UI
# st.title("MBTI Personality Classification")

# tabs = ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]
# questions = [
#     ["What motivates you?", "How do you handle stress?", "Describe your ideal work environment."],
#     ["Do you prefer working alone or in a team?", "How do you make important decisions?", "What is your biggest strength?"],
#     ["How do you react to new challenges?", "Do you enjoy planning ahead?", "Are you more practical or imaginative?"],
#     ["What is your approach to conflict resolution?", "Do you prefer structured or flexible work?", "What are your core values?"],
#     ["How do you define success?", "Do you prefer spontaneity or routine?", "How do you recharge after a long day?"]
# ]

# for i, tab in enumerate(tabs):
#     with st.expander(tab):
#         for j, q in enumerate(questions[i]):
#             key = generate_unique_key(q, i * 10 + j)  # Unique key per question
#             answer = st.text_area(q, key=key)
            
#             if st.button(f"Save Answer {i}-{j}", key=f"save_{i}_{j}"):
#                 if len(answer.strip().split()) < 1:
#                     st.warning("Elongate the answer! Minimum 1 words required.")
#                 elif answer.strip() == "0":
#                     st.info("Skipped question.")
#                     st.session_state.saved_answers[q] = "Skipped."
#                 else:
#                     st.session_state.saved_answers[q] = preprocess_text(answer)
#                     st.success("Answer saved!")

# # Final Submit Button
# if st.button("Submit All Answers"):
#     if len(st.session_state.saved_answers) == 0:
#         st.warning("No answers saved. Please respond to at least one question.")
#     else:
#         # Format as: "Question Answer"
#         final_text = " ".join([f"{q} {preprocess_text(a)}" for q, a in st.session_state.saved_answers.items()])
        
#         # Tokenize and send to model
#         inputs = tokenizer(
#             final_text,
#             max_length=512,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         ).to(device)

#         with torch.no_grad():
#             outputs = model(**inputs)

#         predicted_index = torch.argmax(outputs.logits, dim=1).item()
#         predicted_mbti = MBTI_CLASSES[predicted_index]

#         st.success("Answers submitted successfully! Processing with the model...")
#         # st.write(f"**Processed Text:**\n{final_text}")  # Debugging
#         st.write(f"**Predicted Personality Type:** {predicted_mbti}")  # Final Output


st.set_page_config(page_title="MBTI Personality Test", page_icon="🔮", layout="wide")
st_lottie(lottie_animation, height=300, key="PERSONALITY")

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #4A00E0, #8E2DE2);
            color: white;
        }
        .stApp {
            background-color: #1E1E2F;
            border-radius: 15px;
            padding: 20px;
        }
        .css-1v0mbdj {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #FFC107;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">🔮 Discover Your Personality Type 🔮</p>', unsafe_allow_html=True)
  
tabs = ["💡 Topic 1", "🤔 Topic 2", "🚀 Topic 3", "🧩 Topic 4", "🎭 Topic 5"]
questions = [
    ["What motivates you?", "How do you handle stress?", "Describe your ideal work environment."],
    ["Do you prefer working alone or in a team?", "How do you make important decisions?", "What is your biggest strength?"],
    ["How do you react to new challenges?", "Do you enjoy planning ahead?", "Are you more practical or imaginative?"],
    ["What is your approach to conflict resolution?", "Do you prefer structured or flexible work?", "What are your core values?"],
    ["How do you define success?", "Do you prefer spontaneity or routine?", "How do you recharge after a long day?"]
]

for i, tab in enumerate(tabs):
    with st.expander(tab):
        for j, q in enumerate(questions[i]):
            key = generate_unique_key(q, i * 10 + j)  # Unique key per question
            answer = st.text_area(q, key=key)
            
            if st.button(f"💾 Save Answer", key=f"save_{i}_{j}"):
                if len(answer.strip().split()) < 1:
                    st.warning("⚠️ Elongate the answer! Minimum 1 word required.")
                elif answer.strip() == "0":
                    st.info("⏭️ Skipped question.")
                    st.session_state.saved_answers[q] = "Skipped."
                else:
                    st.session_state.saved_answers[q] = answer
                    st.success("✅ Answer saved!")

# Final Submit Button
if st.button("🚀 Submit All Answers"):
    if len(st.session_state.saved_answers) == 0:
        st.warning("⚠️ No answers saved. Please respond to at least one question.")
    else:
        # Format as: "Question Answer"
        final_text = " ".join([f"{q} {a}" for q, a in st.session_state.saved_answers.items()])
        
        # Tokenize and send to model
        inputs = tokenizer(
            final_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_index = torch.argmax(outputs.logits, dim=1).item()
        predicted_mbti = MBTI_CLASSES[predicted_index]

        st.success("🎉 Answers submitted successfully! Processing with the model...")
        st.write(f"**🔮 Predicted Personality Type:** {predicted_mbti}")


