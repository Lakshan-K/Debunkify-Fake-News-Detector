# Import necessary libraries
import pandas as pd  # Import the pandas library for data manipulation
import numpy as np   # Import the numpy library for numerical operations
import re            # Import the re library for regular expressions
import csv           # Import the csv library for handling CSV files
import random        # Import the random library for generating random values
import string        # Import the string library for string manipulation
import tensorflow as tf  # Import the TensorFlow library for deep learning
import pprint        # Import the pprint library for pretty printing
import os            # Import the os library for interacting with the operating system
import pickle        # Import the pickle library for saving and loading Python objects

# Import specific modules from Keras, a deep learning library
from keras.preprocessing.text import Tokenizer  # Import Tokenizer for text processing
from keras.preprocessing.sequence import pad_sequences  # Import pad_sequences for padding sequences
from keras.utils import to_categorical  # Import to_categorical for one-hot encoding
from keras import regularizers  # Import regularizers for regularization
from keras.models import Sequential  # Import Sequential for creating a sequential model
from keras.layers import Embedding, LSTM, Dense, Dropout  # Import various layers for building the model
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
from sklearn import preprocessing  # Import preprocessing for label encoding
from keras.callbacks import EarlyStopping, ModelCheckpoint  # Import callbacks for model training
from keras.models import load_model  # Import load_model for loading a saved model

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
embedding_dim = 50  # Size of word embeddings
max_length = 54  # Maximum length of sequences
trunc_type = 'post'  # Truncation type for sequences
padding_type = 'post'  # Padding type for sequences
oov_tok = "<OOV>"  # Token for out-of-vocabulary words
training_size = 3000  # Size of the training dataset
test_portion = 0.1  # Percentage of data to use for testing

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
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Generating embeddings matrix for our dataset
embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim))
for word, i in word_index1.items():
    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

# Create a sequential neural network model
model = tf.keras.Sequential([
    # Add an Embedding layer for word embeddings
    tf.keras.layers.Embedding(vocab_size1 + 1, embedding_dim,
                              input_length=max_length, weights=[
                                  embeddings_matrix],
                              trainable=False),
    # Add a Dropout layer with a dropout rate of 0.2 to prevent overfitting
    tf.keras.layers.Dropout(0.2),
    # Add a 1D Convolutional layer with 64 filters, a filter size of 5, and ReLU activation
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    # Add a MaxPooling1D layer with a pool size of 4 to reduce dimensionality
    tf.keras.layers.MaxPooling1D(pool_size=4),
    # Add an LSTM (Long Short-Term Memory) layer with 64 units for sequence processing
    tf.keras.layers.LSTM(64),
    # Add a Dense layer with a single output unit and a sigmoid activation function for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 50

# Convert the training sequences (padded) to a NumPy array and store it in 'training_padded'
training_padded = np.array(training_sequences1)
# Convert the training labels to a NumPy array and store it in 'training_labels'
training_labels = np.array(training_labels)
# Convert the testing sequences (padded) to a NumPy array and store it in 'testing_padded'
testing_padded = np.array(test_sequences1)
# Convert the testing labels to a NumPy array and store it in 'testing_labels'
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