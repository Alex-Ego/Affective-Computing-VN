#####
##  Setup
#####

# Tensorflow libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import string

from tensorflow.keras import layers
import keras

# Text processing tools

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Functions to save models

import os

# Neural Network parameters

vocab_size = 5000 # Size of the vocabulary I'll be using
embedding_dim = 64
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .75

random.seed(96024)

# File setup

import csv

abs_location = os.path.dirname(os.path.abspath(__file__)) # Absolute path leading to the script file
file_location = "/nndata/datasets/" # Path where the data files are
file_name = "text_emotion.csv" # The data file name
    ### The data in the filename is as follows: id, sentiment, author, message
file_name2 = "Jan9-2012-tweets-clean.txt"
    ### The data in the filename is as follows: id:[tab]message[space][tab]::[space]sentiment
file_name3 = ["test.txt", "train.txt", "val.txt"]
    ### The data in the filename is as follows: message;sentiment

# Dumping into vectors
messages = []
labels = []

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

# TESTING -- List of sentiments to append
test_check = ["sadness", "neutral", "happiness", "fun", "worry", "boredom", "joy", "love", "fear"]
# Opening the .csv data file
with open(abs_location + file_location + file_name, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[1] in test_check: # TESTING -- Cutting the size of the sentiments used, REMOVE ME
            if row[1] in ["sadness", "worry", "fear"]:
                labels.append("sadness") # Appending the sentiment associated with the row itself
            elif row[1] in ["neutral", "boredom"]:
                labels.append("neutral") # Appending the sentiment associated with the row itself
            elif row[1] in ["happiness", "fun", "joy", "love"]:
                labels.append("happiness") # Appending the sentiment associated with the row itself
            message = row[3]
            #print("Input: " + message)             #   Debugging purposes
            messages.append(tokenizing_process(message))
with open(abs_location + file_location + file_name2, 'r') as txtfile:
    txtreader = txtfile.readlines()
    for row in txtreader:
        row = row.split("\t")
        row[2] = row[2].strip(" \n") #Cleaning out the unneeded characters
        if row[2] in test_check: # TESTING -- Cutting the size of the sentiments used, REMOVE ME
            if row[2] in ["sadness", "worry", "fear"]:
                labels.append("sadness") # Appending the sentiment associated with the row itself
            elif row[2] in ["neutral", "boredom"]:
                labels.append("neutral") # Appending the sentiment associated with the row itself
            elif row[2] in ["happiness", "fun", "joy", "love"]:
                labels.append("happiness") # Appending the sentiment associated with the row itself
            message = row[1]
            #print("Input: " + message)             #   Debugging purposes
            messages.append(tokenizing_process(message))
""" for f in file_name3:
    with open(abs_location + file_location + "/praveen_emotion_dataset/" + f, 'r') as txtfile:
        txtreader = txtfile.readlines()
        for row in txtreader:
            row = row.split(";")
            row[1] = row[1].strip(" \n") #Cleaning out the unneeded characters
            if row[1] in test_check: # TESTING -- Cutting the size of the sentiments used, REMOVE ME
                if row[1] in ["sadness", "worry", "fear"]:
                    labels.append("sadness") # Appending the sentiment associated with the row itself
                elif row[1] in ["neutral", "boredom"]:
                    labels.append("neutral") # Appending the sentiment associated with the row itself
                elif row[1] in ["happiness", "fun", "joy", "love"]:
                    labels.append("happiness") # Appending the sentiment associated with the row itself
                message = row[0]
                #print("Input: " + message)             #   Debugging purposes
                messages.append(tokenizing_process(message)) """

# Shuffling the data
#print(len(labels)) # Number of labels
#print(len(messages)) # Number of messages
shuffling_var = list(zip(labels, messages))
random.shuffle(shuffling_var)
labels[:], messages[:] = zip(*shuffling_var)
#print(len(labels)) # Number of labels comparison
#print(len(messages)) # Number of messages comparison
dataset = tf.data.Dataset.from_tensor_slices(list(zip(messages, labels)))
##for i in dataset:
##    print(i)

# Training and testing splitting

train_size = int(len(messages) * training_portion)

train_messages = messages[0: train_size]
train_labels = labels[0: train_size]

validation_messages = messages[train_size:]
validation_labels = labels[train_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_messages)
word_index = tokenizer.word_index
# print(dict(list(word_index.items())[0:100]))

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

model = tf.keras.Sequential([
    layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_length),
    layers.SpatialDropout1D(0.15),
    layers.Bidirectional(layers.LSTM(32, dropout=0.15, recurrent_dropout=0.15)),
    #layers.Dense(8, activation="tanh"),
    layers.Dense(4, activation="softmax")
])
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
NUM_EPOCHS = 30
#model.fit(train_padded, training_label_seq, epochs=NUM_EPOCHS)
history = model.fit(train_padded, training_label_seq, epochs=NUM_EPOCHS
 , validation_data=(train_padded, training_label_seq))
model.evaluate(train_padded, training_label_seq)

""" model = tf.keras.Sequential([
    layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_length),
    layers.Conv1D(128, 5, activation="relu"),
    #layers.GlobalAveragePooling1D(),
    #layers.SpatialDropout1D(0.15),
    layers.Bidirectional(layers.LSTM(32, dropout=0.15, recurrent_dropout=0.15)),
    layers.Dense(64, activation="tanh"),
    layers.Dense(4, activation="softmax")
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.summary()

num_epochs = 100
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=1)
loss, accuracy = model.evaluate(train_padded, training_label_seq, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(validation_padded, validation_label_seq, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
 """

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Saving the model and the tokenizer
model_location = os.path.join(abs_location, "nndata/model")
keras.models.save_model(model, model_location + "/sentimental_analysis2.hdf5")
with open(model_location + "/tokens.txt", "w") as f:
    f.write(tokenizer.to_json())
    f.close()

while 1:
    txt = input("Write something: ")
    token_txt = tokenizing_process(txt)
    #print(token_txt)
    separated_token_txt = [token_txt]
    seq = tokenizer.texts_to_sequences(separated_token_txt)
    #print(seq)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded)
    labels = ["sadness", "neutral", "happiness", 0]
    print(pred, labels[np.argmax(pred)])
