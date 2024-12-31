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

class AdvancedVoiceChatbot:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = FrenchStemmer()
        self.load_training_data()
        
    def load_training_data(self):
        """On pourrait Charger et préparer les données d'entraînement avancées du chatbot"""
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
            "météo": {
                "patterns": [
                    "météo", "temps qu'il fait", "température", "climat"
                ],
                "responses": [
                    "Je peux vous donner la météo. Pour quelle ville souhaitez-vous la connaître?"
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
                    1. Vous donner l'heure et la date
                    2. Vous renseigner sur la météo
                    3. Répondre à des questions générales
                    4. Effectuer des calculs simples
                    5. Et bien plus encore! N'hésitez pas à me poser vos questions."""
                ]
            }
        }

    def preprocess_text(self, text):
        """Prétraitement du texte en entrée de manière avancée"""
        try:
            # Conversion en minuscules
            text = text.lower()
            
            # Tokenization
            tokens = self.tokenizer.tokenize(text)
            
            # Stemming (racinalisation) des mots en français
            tokens = [self.stemmer.stem(token) for token in tokens]
            
            return tokens
        except Exception as e:
            st.error(f"Erreur lors du prétraitement: {str(e)}")
            return text.lower().split()

    def analyze_sentiment(self, text):
        """Analyse du sentiment du texte"""
        try:
            blob = TextBlob(text)
            # Convertit le score de sentiment (-1 à 1) en catégorie
            if blob.sentiment.polarity > 0.3:
                return "positif"
            elif blob.sentiment.polarity < -0.3:
                return "négatif"
            return "neutre"
        except:
            return "neutre"

    def get_intent(self, tokens):
        """Détermination de l'intention de l'utilisateur"""
        max_similarity = 0
        matched_intent = None

        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                pattern_tokens = self.preprocess_text(pattern)
                # Calcul de similarité simple basé sur les mots communs
                similarity = len(set(tokens) & set(pattern_tokens)) / len(set(tokens + pattern_tokens))
                if similarity > max_similarity and similarity > 0.2:  # Seuil minimum de similarité
                    max_similarity = similarity
                    matched_intent = intent

        return matched_intent

    def get_weather(self, city):
        """Récupèration de la météo pour une ville donnée (simulation)"""
        weather_data = {
            "Paris": {"temp": "22°C", "condition": "ensoleillé"},
            "Lyon": {"temp": "20°C", "condition": "nuageux"},
            "Marseille": {"temp": "25°C", "condition": "ensoleillé"},
            "Lille": {"temp": "18°C", "condition": "pluvieux"}
        }
        return weather_data.get(city.title(), {"temp": "20°C", "condition": "indisponible"})

    def handle_weather_intent(self, user_input):
        """Gérer les demandes de météo"""
        cities = ["Paris", "Lyon", "Marseille", "Lille"]
        for city in cities:
            if city.lower() in user_input.lower():
                weather = self.get_weather(city)
                return f"À {city}, il fait {weather['temp']} et le temps est {weather['condition']}."
        return "Pour quelle ville souhaitez-vous connaître la météo?"

    def get_response(self, user_input):
        """Générer une réponse avancée en fonction de l'entrée utilisateur"""
        try:
            # Prétraitement
            tokens = self.preprocess_text(user_input)
            
            # Détection de l'intention
            intent = self.get_intent(tokens)
            
            # Analyse du sentiment
            sentiment = self.analyze_sentiment(user_input)
            
            # Si l'intention est météo
            if "météo" in user_input.lower():
                return self.handle_weather_intent(user_input)
            
            # Si une intention est détectée
            if intent and intent in self.intents:
                response = random.choice(self.intents[intent]["responses"])
                # Si la réponse est une fonction
                if callable(response):
                    return response()
                return response
            
            # Réponse par défaut selon le sentiment
            if sentiment == "négatif":
                return "Je sens que quelque chose vous préoccupe. Comment puis-je vous aider?"
            elif sentiment == "positif":
                return "Je suis ravi de voir votre enthousiasme! Comment puis-je vous assister?"
            
            return "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ou me dire plus précisément ce que vous souhaitez?"
            
        except Exception as e:
            st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
            return "Désolé, une erreur s'est produite. Pouvez-vous réessayer?"

    def transcribe_speech(self):
        """Transcription de la parole en texte avec gestion avancée des erreurs"""
        try:
            with sr.Microphone() as source:
                # Ajustement pour le bruit ambiant
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                st.write("🎤 En écoute... Parlez maintenant!")
                
                # Paramètres avancés pour la reconnaissance
                audio = self.recognizer.listen(source, 
                                             timeout=5,
                                             phrase_time_limit=10)
                
                st.write("📝 Traitement de votre message...")
                
                # Tentative avec le français d'abord
                try:
                    text = self.recognizer.recognize_google(audio, language='fr-FR')
                except:
                    # Fallback en anglais si le français échoue
                    text = self.recognizer.recognize_google(audio, language='en-US')
                
                return text
                
        except sr.WaitTimeoutError:
            return "⚠️ Aucune parole détectée. Veuillez réessayer."
        except sr.UnknownValueError:
            return "⚠️ Désolé, je n'ai pas compris. Pourriez-vous répéter?"
        except sr.RequestError:
            return "⚠️ Service de reconnaissance vocale temporairement indisponible."
        except Exception as e:
            return f"⚠️ Une erreur est survenue: {str(e)}"

def load_custom_css():
    """Charge le CSS personnalisé pour l'interface"""
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            border: none;
            font-weight: bold;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 10px;
            border-radius: 10px;
            margin
        .user-message {
            background-color: #e6f3ff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: left;
        }
        .bot-message {
            background-color: #f2f2f2;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Fonction principale pour l'application Streamlit"""
    st.title("Assistant Vocal Intelligent")
    st.write("Bienvenue! Cliquez sur le bouton pour commencer à parler à votre assistant.")
    
    # Charger le CSS personnalisé
    load_custom_css()
    
    # Créer une instance du chatbot
    chatbot = AdvancedVoiceChatbot()
    
    # Section d'entrée audio
    if st.button("🎤 Parlez maintenant"):
        # Transcription de la parole en texte
        user_input = chatbot.transcribe_speech()
        
        if user_input:
            st.write(f"🗣️ Vous avez dit: {user_input}")
            
            # Générer la réponse du chatbot
            response = chatbot.get_response(user_input)
            st.write(f"🤖 Réponse: {response}")
            
            # Affichage des messages utilisateur et bot dans un format agréable
            st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)

# Exécution de l'application Streamlit
if __name__ == "__main__":
    main()
