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


CAREER_PATHS = {
    "INTJ": ["Software Developer/Engineer", "Investment Analyst / Financial Strategist", "Startup Founder / CEO", "Lawyer", "Policy Analyst", "Research Scientist"],
    "INTP": ["Writer", "Professor/Researcher", "Philosopher", "Forensic Scientist", "Astronomer", "Research Scientist"],
    "ENTJ": ["Entrepreneur", "Investment Banker", "Politician", "Military Officer", "Film Director", "Civil Servant"],
    "ENTP": ["Journalist/Investigative Reporter", "Public Speaker", "Podcast Host/Content Creator", "Venture Capitalist", "Tech Startup Founder", "Marketing Strategist"],
    "INFJ": ["Life Coach", "Religious Leader", "Social Worker", "Writer", "Journalist", "Politician"],
    "INFP": ["Counselor/Therapist", "Teacher", "Writer", "Content Creator", "Life Coach", "Musician"],
    "ENFJ": ["CEO/Executive", "Career Counselor", "Politician", "Diplomat", "Human Resources Manager", "Teacher/Professor"],
    "ENFP": ["Graphic Designer", "Blogger/Content Creator", "Researcher", "Interior Designer", "Travel Blogger/Photographer", "Journalist"],
    "ISTJ": ["Accountant", "Lawyer/Judge", "Police Officer/Detective", "Pharmacist", "Banker", "Military Officer"],
    "ISFJ": ["Doctor", "Teacher", "Therapist / Mental Health Counselor", "Human Resources Specialist", "Customer Service Representative", "Bank Teller / Loan Officer"],
    "ESTJ": ["CEO / Business Executive", "Bank Manager", "Lawyer / Judge", "Politician / Civil Servant", "Marketing Director", "Mechanical Engineer"],
    "ESFJ": ["Dietitian / Nutritionist", "Doctor", "Paralegal / Legal Assistant", "Social Worker", "Wedding Planner", "Broadcast Journalist"],
    "ISTP": ["Engineer", "Cybersecurity Analyst", "Military Specialist", "Private Investigator", "Pilot", "Race Car Driver"],
    "ISFP": ["Wildlife Photographer", "Zoologist / Animal Caretaker", "Actor / Dancer / Musician", "Chef", "Yoga Instructor", "Artist / Designer"],
    "ESTP": ["Entrepreneur", "Sales Executive", "Police Officer / Detective", "TV Host / News Anchor", "Professional Athlete", "Bartender / Hospitality Manager"],
    "ESFP": ["Actor / TV Personality", "Fashion Designer / Stylist", "Photographer / Videographer", "Flight Attendant", "Event Planner / Wedding Planner", "Comedian / Entertainer"]
}

PERSONALITY_DESCRIPTIONS = {
    "INTJ": "INTJs are strategic thinkers who excel at planning and executing complex ideas. They are independent and driven, often pursuing leadership roles.",
    "INTP": "INTPs are highly analytical and curious, always questioning and exploring new ideas. They are natural problem-solvers and enjoy intellectual challenges.",
    "ENTJ": "ENTJs are natural leaders who thrive in high-pressure environments. They are decisive, assertive, and excel at organizing and managing people and resources.",
    "ENTP": "ENTPs are energetic and love engaging in intellectual debates. They are quick thinkers and thrive in dynamic environments that encourage creativity and innovation.",
    "INFJ": "INFJs are insightful and empathetic, always striving to help others. They have strong values and a deep sense of purpose in their work and relationships.",
    "INFP": "INFPs are idealistic and value deep personal connections. They are creative and driven by their passion for making a positive impact in the world.",
    "ENFJ": "ENFJs are charismatic and persuasive leaders. They are deeply empathetic and skilled at understanding the needs of others, making them excellent mentors and motivators.",
    "ENFP": "ENFPs are enthusiastic and spontaneous, always seeking new experiences. They are highly creative and excel in roles that allow for self-expression and freedom.",
    "ISTJ": "ISTJs are practical and highly organized. They value tradition, responsibility, and attention to detail, making them reliable and hardworking professionals.",
    "ISFJ": "ISFJs are compassionate and dedicated individuals who take pride in helping others. They are highly dependable and excel in caregiving professions.",
    "ESTJ": "ESTJs are strong-willed and thrive in structured environments. They are natural managers who excel at enforcing rules and maintaining order.",
    "ESFJ": "ESFJs are warm and sociable, always eager to support those around them. They thrive in roles that involve community engagement and service.",
    "ISTP": "ISTPs are highly resourceful and love hands-on work. They excel in problem-solving and enjoy exploring how things work.",
    "ISFP": "ISFPs are artistic and deeply in touch with their emotions. They seek beauty in the world and express themselves through various creative outlets.",
    "ESTP": "ESTPs are adventurous and thrive in fast-paced environments. They are highly energetic and enjoy taking risks in both business and life.",
    "ESFP": "ESFPs are vibrant and full of life, always seeking fun and excitement. They are natural performers and excel in entertainment and social professions."
}

lottie_animation = load_lottieurl("https://lottie.host/7668124b-6b16-44c9-831b-62878be9ce9c/SI794Nk3RM.json")
lottie_animation_2 = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_w51pcehl.json")

# Initialize session state for saved answers
if "saved_answers" not in st.session_state:
    st.session_state.saved_answers = {}

st.set_page_config(page_title="Personality Test For Career Path", page_icon="🔮", layout="wide")


st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #2C2E43, #E03B8B);
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
        .stLottie {
            background: none !important;
            mix-blend-mode: screen; /* Optional: Blends with background */
        }
        .lottie-container {
            width: 250px;
            height: 150px;
            background-color: #1E1E2F; /* Light gray */
            border-radius: 10px;
            padding: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: auto; /* Center it */
        }
        .center-button {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

col, col2 = st.columns([1,1])
st_lottie(lottie_animation, speed=1, height=300, key="personality")

st.markdown('<p class="title">🔮 Discover Your Career Path Based On Your Personality Type 🔮</p>', unsafe_allow_html=True)
  
tabs = ["💡 Topic 1-Motivation & Work Ethic", "🤔 Topic 2-Work Style & Preferences", "🚀 Topic 3-Decision-Making & Problem-Solving", "🧩 Topic 4-Stress & Conflict Management", "🎭 Topic 5-Planning & Adaptability"]
questions = [
    ["What motivates you?", "What is your biggest strength?", "How do you define success?","What are your core values?"],
    ["Describe your ideal work environment.", "Do you prefer working alone or in a team?", "Do you prefer structured or flexible work?","Do you prefer spontaneity or routine?"],
    ["How do you make important decisions?", "How do you react to new challenges?", "Are you more practical or imaginative?"],
    ["How do you handle stress?", "What is your approach to conflict resolution?", "How do you recharge after a long day?"],
    ["Do you enjoy planning ahead?", "How do you deal with obstacles that come under your way", "Do you adapt to new environment quickly or take time"]
]


for i, tab in enumerate(tabs):
    with st.expander(tab):
        for j, q in enumerate(questions[i]):
            key = generate_unique_key(q, i * 10 + j)  # Unique key per question
            answer = st.text_area(q, key=key)
            
            if st.button(f"💾 Save Answer", key=f"save_{i}_{j}"):
                if len(answer.strip().split()) < 1:
                    st.warning("⚠️ Please eneter atleast 1 word. If u wish to skip then enter 0")
                elif answer.strip() == "0":
                    st.info("⏭️ Skipped question.")
                    st.session_state.saved_answers[q] = "Skipped."
                else:
                    st.session_state.saved_answers[q] = answer
                    st.success("✅ Answer saved!")
                  
st_lottie(lottie_animation_2, speed=1, height=100, key="good")

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
        st.write(f"**🧠 Personality Traits:** {PERSONALITY_DESCRIPTIONS[predicted_mbti]}")
        st.write("**💼 Appropriate Career Paths:**")
        for career in CAREER_PATHS[predicted_mbti]:
            st.write(f"- {career}")


st_lottie(load_lottieurl("https://lottie.host/4a73f884-4430-4b65-ab81-6f90dd3fa8a4/DtKeLhEmuX.json"), width=150, height=100)

