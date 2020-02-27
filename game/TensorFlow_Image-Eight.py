
########## Second attempt at actually using TensorFlow, let's go.

#####
##  Setup
#####

# Tensorflow libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras import layers

# Text processing tools

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# My local functions that I've used previously

import datadict as dd

# Neural Network parameters

vocab_size = 8000 # Size of the vocabulary I'll be using
embedding_dim = 64
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .7


# File setup

import csv
import os

abs_location = os.path.dirname(os.path.abspath(__file__)) # Absolute path leading to the script file
file_location = "/nndata/datasets/" # Path where the data files are
file_name = "text_emotion.csv" # The data file name
    ### The data in the filename is as follows: id, sentiment, author, message

# Dumping into vectors

raw_data = []
messages = []
labels = []

# Opening the .csv data file
with open(abs_location + file_location + file_name, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        labels.append(row[1]) # Appending the sentiment associated with the row itself
        message = row[3]
        # Filtering the stopwords, which are the most commonly used words in the English language, like prepositions and stuff like that.
        for word in STOPWORDS:
            token = ' ' + word + ' '
            message = message.replace(token, ' ')
            message = message.replace(' ', ' ')
        messages.append(message)

# Training and testing splitting

train_size = int(len(messages) * training_portion)

train_messages = messages[0: train_size]
train_labels = labels[0: train_size]

validation_messages = messages[train_size:]
validation_labels = labels[train_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_messages)
word_index = tokenizer.word_index

# Making lists of tokens

train_sequences = tokenizer.texts_to_sequences(train_messages)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_messages)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# Building the model

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_length))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(20, activation="sigmoid"))
model.add(layers.Dense(14, activation="sigmoid"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 50
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
loss, accuracy = model.evaluate(train_padded, training_label_seq, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(validation_padded, validation_label_seq, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

while 1:
    text = input("Write something: ")
    txt = []
    txt.append(text)
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded)
    labels = ["anger", "boredom", "empty", "enthusiasm", "fun", "happiness", "hate", "love", "neutral", "relief", "sadness", "surprise", "worry", "N/A"]
    print(pred, labels[np.argmax(pred)])
