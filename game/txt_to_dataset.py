
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

from tensorflow.keras import layers

# Text processing tools

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# My local functions that I've used previously

import datadict as dd

# File setup

import os

abs_location = os.path.dirname(os.path.abspath(__file__)) # Absolute path leading to the script file
file_location = "/nndata/test/" # Path where the data text files are
file_name_list = ["testdata.txt", "traindata.txt"] # The text file names

# Dumping into vectors

raw_data = []
tweets = []
labels = []

for file_name in file_name_list:
    raw_data = dd.datadump(abs_location + file_location + file_name)

for tweet in raw_data:
    tweets.append(tweet[0])
    labels.append(tweet[1])

print(tweets)

#####
##  Encoding lines as numbers
#####

###Build vocabulary
#tokenizer = tfds.features.text.Tokenizer()

#vocabulary_set = set()
#for text_tensor, _ in all_labeled_data:
#    some_tokens = tokenizer.tokenize(text_tensor.numpy())
#    vocabulary_set.update(some_tokens)

#vocab_size = len(vocabulary_set)


