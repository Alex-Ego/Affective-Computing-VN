import datadict as dd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

while 1:
    txt = input("Write something: ")
    filtered = dd.clean_data(txt)
