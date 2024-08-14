import streamlit as st
import requests
from bs4 import BeautifulSoup
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Ignore warnings
warnings.filterwarnings('ignore')


# Read the text file
corpus_file = 'chatbot.txt'

def read_corpus():
    with open(corpus_file, 'r', errors='ignore') as f:
        raw = f.read()
    return raw

def update_corpus(new_data):
    with open(corpus_file, 'a', encoding='utf-8') as f:
        f.write('\n' + new_data)

# Convert to lowercase
raw_corpus = read_corpus()
raw = raw_corpus.lower()

# Tokenize the text
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Lemmatization
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you"
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

# Function to perform web search and scrape content
def perform_web_search(query):
    try:
        # Perform web search
        url = f"https://www.google.com/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse search results using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='tF2Cxc')
        
        if search_results:
            # Extract and return content from the first result
            content = search_results[0].get_text()
            update_corpus(content)  # Update corpus with new content
            return content
        else:
            return None
    except Exception as e:
        st.error(f"Error performing web search: {e}")
        return None

# Streamlit GUI
st.title("ChatBot")

if 'conversation' not in st.session_state:
    st.session_state.conversation = ""

user_input = st.text_input("You: ", "")

if st.button("Send"):
    user_response = user_input.lower()
    st.session_state.conversation += f"You: {user_response}\n"

    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            reply = "You are welcome.."
        else:
            if greeting(user_response) is not None:
                reply = greeting(user_response)
            else:
                reply = response(user_response)
                if reply.startswith("I am sorry!"):
                    web_content = perform_web_search(user_response)
                    if web_content:
                        reply = web_content

        st.session_state.conversation += f"ROBO: {reply}\n"
    else:
        st.session_state.conversation += "ROBO: Bye! take care..\n"

st.text_area("Conversation", st.session_state.conversation, height=400)
