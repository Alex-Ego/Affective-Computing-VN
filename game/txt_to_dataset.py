
########## First attempt at actually using TensorFlow, let's see how this goes.

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

vocab_size = 5000 # Size of the vocabulary I'll be using
embedding_dim = 64
max_length = 20
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8


# File setup

import csv
import os

abs_location = os.path.dirname(os.path.abspath(__file__)) # Absolute path leading to the script file
file_location = "/nndata/datasets/" # Path where the data files are
file_name_list = ["joy.csv", "love.csv", "anger.csv", "sadness.csv"] # The data files' names
    ### The data in the filename is as follows: id, yes or no depending on the dataset we're in, message

# Dumping into vectors

raw_data = []
messages = []
labels = []

for file_name in file_name_list:
    # Getting the correct name for the label, inferred from the name of the data file
    label_name = list(file_name)
    for i in range(4):
        label_name.pop()
    label_name = "".join(label_name)
    # Opening the .csv data file
    with open(abs_location + file_location + file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[1] == "YES":
                labels.append(label_name) # Appending the name of the file instead of using the raw "yes/no" format
                message = row[2]
                for word in STOPWORDS:
                    token = ' ' + word + ' '
                    message = message.replace(token, ' ')
                    message = message.replace(' ', ' ')
                messages.append(message)

# Shuffling lists

shuffled_data = list(zip(messages, labels))
random.seed(150506)
random.shuffle(shuffled_data)
messages[:], labels[:] = zip(*shuffled_data)


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


#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#def decode_message(text):
#    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#for i in range(len(train_padded)):
#    print(decode_message(train_padded[i]))
#    print('---')
#    print(train_messages[i])
#    print()

model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

text = input("Write something: ")
txt = []
txt.append(text)
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
labels = ["anger", "joy", "love", "sadness"]
print(pred, labels[np.argmax(pred)])
