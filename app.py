import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Načteme model a tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)


# Funkce pro zobrazení sentimentu s použitím smajlíků
def display_sentiment(sentiment):
    sentiment_emojis = {
        "Positive": "😊",
        "Neutral": "😐",
        "Negative": "😞"
    }
    st.markdown(f"<h3>{sentiment_emojis[sentiment]} Sentiment: {sentiment}</h3>", unsafe_allow_html=True)

# Výsledky analýzy
if st.button("Analyzovat sentiment"):
    if user_input:
        sentiment = analyze_sentiment(user_input)
        display_sentiment(sentiment)
    else:
        st.write("Prosím, zadejte text recenze pro analýzu.")


# Streamlit rozhraní
st.title("Sentiment Analysis of Product Reviews")
st.write("Zadejte recenzi a zjistěte, zda je pozitivní, negativní, nebo neutrální.")

# Vstupní text od uživatele
user_input = st.text_area("Zadejte recenzi produktu:")

# Tlačítko pro analýzu
if st.button("Analyzovat sentiment"):
    if user_input:
        sentiment = analyze_sentiment(user_input)
        st.write(f"Sentiment této recenze je: {sentiment}")
    else:
        st.write("Prosím, zadejte text recenze pro analýzu.")

