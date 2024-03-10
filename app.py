import streamlit as st
import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from keras.models import load_model

# Download NLTK resources
 nltk.download('wordnet')
# Load pre-trained model and associated data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents (1).json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words) 
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict class/intent
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get response based on predicted intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Main function
def main():
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # App title and icon
    st.title("E-COMMERCE CHATBOT")
    st.image("chatbot_icon.png", width=100)

    # Input message
    message = st.text_input("You: ")

    # Send button
    if st.button("Send"):
        if message:
            # Predict intent and get response
            ints = predict_class(message, model)
            response = get_response(ints, intents)

            # Display response
            st.text_area("Bot:", value=response, height=200, max_chars=None, key=None)

            # Store message and response in chat history
            st.session_state['chat_history'].append({"user": message, "bot": response})

    # Display chat history
    if st.session_state['chat_history']:
        st.subheader("Chat History")
        for entry in st.session_state['chat_history']:
            st.text_area("User:", value=entry["user"], height=100, max_chars=None, key=None)
            st.text_area("Bot:", value=entry["bot"], height=100, max_chars=None, key=None)

if __name__ == "__main__":
    main()
