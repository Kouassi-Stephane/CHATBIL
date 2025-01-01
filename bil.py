import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import FrenchStemmer
import string
import random
import json
import datetime
import requests
import pandas as pd
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Assistant Vocal Intelligent",
    page_icon="🤖",
    layout="centered"  # Changé de "wide" à "centered" pour plus de stabilité
)

# CSS sécurisé pour Streamlit
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            padding: 0.5em 1em;
            border-radius: 5px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 0.5em;
            border-radius: 5px;
            margin: 0.5em 0;
        }
        .bot-message {
            background-color: #f0f0f0;
            padding: 0.5em;
            border-radius: 5px;
            margin: 0.5em 0;
        }
    </style>
""", unsafe_allow_html=True)

class AdvancedVoiceChatbot:
    def _init_(self):
        self.recognizer = sr.Recognizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = FrenchStemmer()
        self.load_training_data()
        
    def load_training_data(self):
        """Charge et prépare les données d'entraînement du chatbot"""
        self.intents = {
            "salutations": {
                "patterns": [
                    "bonjour", "salut", "hello", "coucou", "hey", "bonsoir",
                    "comment ça va", "comment vas-tu", "ça va"
                ],
                "responses": [
                    "Bonjour! Je suis votre assistant virtuel. Comment puis-je vous aider?",
                    "Salut! Ravi de vous parler. Que puis-je faire pour vous?",
                    "Hey! Je suis là pour vous aider. Que souhaitez-vous faire?"
                ]
            },
            "aurevoir": {
                "patterns": [
                    "au revoir", "bye", "à bientôt", "à plus", "adieu", "bonne journée"
                ],
                "responses": [
                    "Au revoir! Passez une excellente journée!",
                    "À bientôt! N'hésitez pas à revenir si vous avez besoin d'aide.",
                    "Au revoir et merci de votre visite!"
                ]
            },
            "heure": {
                "patterns": [
                    "quelle heure", "l'heure", "heure actuelle", "temps"
                ],
                "responses": [
                    lambda: f"Il est actuellement {datetime.datetime.now().strftime('%H:%M')}."
                ]
            },
            "date": {
                "patterns": [
                    "quel jour", "quelle date", "date aujourd'hui", "on est quel jour"
                ],
                "responses": [
                    lambda: f"Nous sommes le {datetime.datetime.now().strftime('%d/%m/%Y')}."
                ]
            },
            "humeur": {
                "patterns": [
                    "comment te sens tu", "ça va", "tu vas bien", "ton humeur"
                ],
                "responses": [
                    "Je suis un programme, mais j'apprécie votre intérêt! Je suis là pour vous aider.",
                    "Très bien, merci! Comment puis-je vous assister aujourd'hui?",
                    "En pleine forme et prêt à vous aider!"
                ]
            },
            "capacités": {
                "patterns": [
                    "que sais tu faire", "tes capacités", "peux tu faire",
                    "tes fonctionnalités", "aide", "help"
                ],
                "responses": [
                    """Je peux vous aider avec plusieurs choses :
                    - Vous donner l'heure et la date
                    - Répondre à des questions générales
                    - Effectuer des calculs simples
                    - Et bien plus encore! N'hésitez pas à me poser vos questions."""
                ]
            }
        }

    def preprocess_text(self, text):
        """Prétraite le texte en entrée"""
        try:
            text = text.lower()
            tokens = self.tokenizer.tokenize(text)
            tokens = [self.stemmer.stem(token) for token in tokens]
            return tokens
        except Exception as e:
            st.error(f"Erreur lors du prétraitement: {str(e)}")
            return text.lower().split()

    def get_intent(self, tokens):
        """Détermine l'intention de l'utilisateur"""
        max_similarity = 0
        matched_intent = None

        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                pattern_tokens = self.preprocess_text(pattern)
                similarity = len(set(tokens) & set(pattern_tokens)) / len(set(tokens + pattern_tokens))
                if similarity > max_similarity and similarity > 0.2:
                    max_similarity = similarity
                    matched_intent = intent

        return matched_intent

    def get_response(self, user_input):
        """Génère une réponse en fonction de l'entrée utilisateur"""
        try:
            tokens = self.preprocess_text(user_input)
            intent = self.get_intent(tokens)
            
            if intent and intent in self.intents:
                response = random.choice(self.intents[intent]["responses"])
                if callable(response):
                    return response()
                return response
            
            return "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler?"
            
        except Exception as e:
            st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
            return "Désolé, une erreur s'est produite. Pouvez-vous réessayer?"

    def transcribe_speech(self):
        """Transcrit la parole en texte"""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                st.write("🎤 En écoute... Parlez maintenant!")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                st.write("📝 Traitement de votre message...")
                
                try:
                    text = self.recognizer.recognize_google(audio, language='fr-FR')
                except:
                    text = self.recognizer.recognize_google(audio, language='en-US')
                
                return text
                
        except sr.WaitTimeoutError:
            return "⚠ Aucune parole détectée. Veuillez réessayer."
        except sr.UnknownValueError:
            return "⚠ Désolé, je n'ai pas compris. Pourriez-vous répéter?"
        except sr.RequestError:
            return "⚠ Service de reconnaissance vocale temporairement indisponible."
        except Exception as e:
            return f"⚠ Une erreur est survenue: {str(e)}"

def main():
    st.title("🤖 Assistant Vocal Intelligent")
    
    # Introduction
    st.markdown("""
        Je suis votre assistant virtuel intelligent. Je peux vous aider avec :
        - 🗣 Reconnaissance vocale en français
        - ⏰ Information sur l'heure et la date
        - 💬 Conversation naturelle
    """)
    
    # Initialisation du chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AdvancedVoiceChatbot()
    
    # Initialisation de l'historique des messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Interface utilisateur simplifiée
    input_mode = st.radio("Mode d'entrée:", ("Texte", "Voix"))
    
    if input_mode == "Texte":
        if prompt := st.chat_input("Écrivez votre message ici..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            response = st.session_state.chatbot.get_response(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
    
    else:
        if st.button("🎤 Cliquez pour parler"):
            user_input = st.session_state.chatbot.transcribe_speech()
            
            if not user_input.startswith("⚠"):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                
                response = st.session_state.chatbot.get_response(user_input)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            else:
                st.error(user_input)

if __name__ == "_main_":
    main()