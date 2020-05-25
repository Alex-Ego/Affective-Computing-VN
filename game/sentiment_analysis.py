
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import string

# Text processing tools

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.models import load_model


abs_location = os.path.dirname(os.path.abspath(__file__))

model = load_model(abs_location + "/nndata/model/sentimental_analysis.hdf5")

vocab_size = 5000 # Size of the vocabulary I'll be using
embedding_dim = 64
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

def tokenizing_process(message):
    # Pre-tokenizing
    tokens = word_tokenize(message)
    # Making them lowercase
    tokens = [w.lower() for w in tokens]
    # Filtering the punctuations
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # Filtering non-alphabetic characters
    words = [word for word in stripped if word.isalpha()]
    # Removing stopwords
    #stop_words = dd.stopwords                      #Custom stopwords
    stop_words = set(stopwords.words('english'))    #Premade stopwords
    words = [w for w in words if not w in stop_words]
    # Stemming words (test)
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    # Joining the resulting string
    message = " ".join(stemmed)
    #print("Output: " + message + "\n")     #   Debugging purposes
    return message

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

def evaluation(txt):
    token_txt = tokenizing_process(txt)
    #print(token_txt)
    separated_token_txt = [token_txt]
    seq = tokenizer.texts_to_sequences(separated_token_txt)
    #print(seq)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded)
    labels = ["sadness", "neutral", "happiness"]
    print(pred, labels[np.argmax(pred)])
    
evaluation("Hello, I'm happy to be here")