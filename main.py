import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
## word embeddingb representation
from tensorflow.keras.datasets import imdb

from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
## mapping of word index back to words
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={value: key for key , value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=100)
    return padded_review 

## step:3 prediction function
## prediction function
def predict_sentiment(review):
    preproces_review=preprocess_text(review)
    prediction=model.predict(preproces_review)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment, prediction[0][0]

## streamlit app
import streamlit as st
st.title('IMDB movie Revie SENTIMENT ANALYSIS')
st.write('Enter a movier review to classify it as positive or negative')

user_input=st.text_area('Movie Review')
if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    decoded_review = decode_review(preprocess_input[0])
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.65 else 'Negative'

    st.write(f'Original Review: {decoded_review}')
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')   


