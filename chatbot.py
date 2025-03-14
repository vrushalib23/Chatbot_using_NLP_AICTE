import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Custom Streamlit Styling
st.markdown(
    """
    <style>

        body {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Poppins', sans-serif;
        }

         
        .stApp {
            background: linear-gradient(135deg, #161b22, #1f6feb);
            color: #c9d1d9;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .chat-container {
            background: rgba(36, 41, 46, 0.9);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.5);
            margin-bottom: 15px;
            transition: all 0.3s ease-in-out;
            width: 60%;
            text-align: center;
        }
        .chat-container:hover {
            transform: translateY(-5px);
        }
        .user-message {
            text-align: right;
            color: #ffffff;
            background: #238636;
            padding: 12px;
            border-radius: 12px;
            margin: 5px 0;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(35, 134, 54, 0.5);
        }
        .bot-message {
            text-align: left;
            color: #ffffff;
            background: #1f6feb;
            padding: 12px;
            border-radius: 12px;
            margin: 5px 0;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(31, 111, 235, 0.5);
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .stSidebar {

            

            width: 110px;
            color : black;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.3);
        }
        .stTitle {
         
            font-size: 2rem;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load intents from the JSON file
file_path = "intents.json"
if not os.path.exists(file_path):
    st.error("Error: 'intents.json' file not found.")
    st.stop()

with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent.get('patterns', []):
        tags.append(intent['tag'])
        patterns.append(pattern)

if patterns:
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)
else:
    st.error("Error: No patterns found in 'intents.json'.")
    st.stop()

def chatbot(input_text):
    if not input_text.strip():
        return "Please enter a valid message."
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

st.sidebar.title("💬 Chatbot Menu")
menu = ["Home", "Chat History", "About"]
choice = st.sidebar.radio("Navigate", menu)

if choice == "Home":
    st.title("🤖 AI Chatbot")
    st.markdown("### Welcome! Type your message below:")
    
    chat_history = st.container()
    with chat_history:
        user_input = st.text_input("You:", "", key="input")
        if user_input:
            response = chatbot(user_input)
            st.markdown(f'<div class="chat-container"><div class="user-message">{user_input}</div><div class="bot-message">{response}</div></div>', unsafe_allow_html=True)

elif choice == "Chat History":
    st.title("📜 Conversation History")
    if os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "r", encoding="utf-8") as file:
            history = csv.reader(file)
            for row in history:
                if len(row) == 3:
                    st.markdown(f'<div class="chat-container"><div class="user-message">User: {row[0]}</div><div class="bot-message">Chatbot: {row[1]}</div></div>', unsafe_allow_html=True)
    else:
        st.warning("No conversation history found.")

elif choice == "About":
    st.title("ℹ️ About This Chatbot")
    st.write("This chatbot uses NLP and Logistic Regression to respond intelligently to user queries.")
    st.write("It features a modern UI with an interactive chat experience and animations.")
    st.write("--VSB")