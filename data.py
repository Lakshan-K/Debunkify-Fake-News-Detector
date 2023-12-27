# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
import re            # For regular expressions
import json          # For JSON handling
import csv           # For CSV handling
import random        # For generating random values
import string        # For string manipulation
import tensorflow as tf  # For creating and training deep learning models
import pprint        # For pretty printing
import os            # For interacting with the operating system
import pickle        # For saving and loading Python objects

# Import specific modules from Keras, a deep learning library
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

# Reading the data from a CSV file named "news.csv" using pandas
data = pd.read_csv("news.csv")

# Display the first few rows of the dataset
data.head()

# Drop a column named "Unnamed: 0" from the dataset
data = data.drop(["Unnamed: 0"], axis=1)

# Display the first 5 rows of the dataset after dropping the column
data.head(5)

# Encode the labels in the 'label' column to numerical values
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])

# Define some parameters for text processing and model training
embedding_dim = 50
max_length = 54
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = 0.1

# Create empty lists to store titles, text, and labels
title = []
text = []
labels = []

# Iterate through the first 3000 rows of the dataset
for x in range(training_size):
    title.append(data['title'][x])
    text.append(data['text'][x])
    labels.append(data['label'][x])

# Create a tokenizer for processing the title text
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)

# Pad the sequences to make them the same length
padded1 = pad_sequences(
    sequences1, padding=padding_type, truncating=trunc_type)

# Split the data into training and test sets
split = int(test_portion * training_size)
training_sequences1 = padded1[split:training_size]
test_sequences1 = padded1[0:split]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

# Create a dictionary to store word embeddings
embeddings_index = {}

# Open and read pre-trained word embeddings from a file
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:]
, dtype='float32')
        embeddings_index[word] = coefs

# Generating embeddings matrix for our dataset
embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim))
for word, i in word_index1.items():
    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

# Create a sequential neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size1+1, embedding_dim,
                              input_length=max_length, weights=[
                                  embeddings_matrix],
                              trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Define the number of training epochs
num_epochs = 50

# Prepare the training and testing data
training_padded = np.array(training_sequences1)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences1)
testing_labels = np.array(test_labels)

# Train the model on the training data
history = model.fit(training_padded, training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded,
                                     testing_labels),
                    verbose=2)

# Save the trained model to a file
model.save("news_model.h5")

# Save the tokenizer to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer1, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the saved model from a file
model = load_model("news_model.h5")

# Load the saved tokenizer from a file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

# Get user input for a news article text
user_input_text = input("Enter the news article text: ")

# Preprocess the user input text
sequences = tokenizer1.texts_to_sequences([user_input_text])[0]
sequences = pad_sequences([sequences], maxlen=54,
                          padding=padding_type,
                          truncating=trunc_type)

# Make a prediction using the trained model
prediction = model.predict(sequences, verbose=0)[0][0]

# Display the result based on the prediction
if prediction >= 0.5:
    print("This news is likely true.")
else:
    print("This news is likely false.")