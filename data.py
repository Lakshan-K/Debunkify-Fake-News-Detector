# Import necessary libraries and modules for data preprocessing, modeling, and evaluation
import pandas as pd             # For data manipulation and handling
import numpy as np              # For numerical operations
import re                       # For regular expressions used in text cleaning
import string                   # For working with string operations
import tensorflow as tf         # For using TensorFlow backend for Keras
import pprint                   # For pretty printing
from keras.preprocessing.text import Tokenizer     # For text tokenization
from keras.preprocessing.sequence import pad_sequences  # For padding sequences
from keras.utils import to_categorical              # For one-hot encoding of labels
from keras import regularizers                      # For regularization in neural networks
from keras.models import Sequential                # For creating sequential neural network models
from keras.layers import Embedding, LSTM, Dense, Dropout  # For specifying layers in the neural network
from sklearn.model_selection import train_test_split  # For splitting the dataset into train and test sets
from sklearn import preprocessing                  # For data preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint  # For setting up callbacks for model training
from keras.models import load_model                # For loading the Keras model
import json 
import os
import pickle

# Define the text cleaning function 
def clean_text(text):
    # Convert the text to lowercase to ensure uniformity
    text = text.lower()
    # Remove text within square brackets, often used for annotations or references
    text = re.sub(r'\[.*?\]', '', text)
    # Replace non-word characters and symbols with spaces
    text = re.sub(r'\W', ' ', text)
    # Remove URLs starting with 'http' or 'https' and 'www' links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove text within angle brackets, which might represent HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation marks using the string.punctuation list
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) 
    # Exclude the word 'in' (considered a stopword) from the text
    text = re.sub(r'in', '', text)
    # Remove alphanumeric words (words containing digits)
    text = re.sub(r'\w*\d\w*', '', text)
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')
    # Return the cleaned text
    return text

fake_csv_path = "news_datasets/Fake.csv"
true_csv_path = "news_datasets/True.csv"

# Read the CSV files
fake_data = pd.read_csv(fake_csv_path)
true_data = pd.read_csv(true_csv_path)

# Assign labels to dataset
fake_data["label"] = 0 
true_data['label'] = 1

# Merge datasets
data_merge = pd.concat([fake_data, true_data], axis=0) 

# Drop columns
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Define a function to preprocess the data
def preprocess_data(data, max_words, maxlen):
    # Clean the text data
    data['text'] = data['text'].apply(clean_text)
    
    # Text Tokenization
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data['text'])
    
    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(data['text'])
    
    # Padding sequences (adjust maxlen as needed)
    X = pad_sequences(sequences, maxlen=maxlen)
    
    # Labels
    y = data['label'].values
    
    return X, y, tokenizer

# Text Tokenization and Padding (only if preprocessed data does not exist)
max_words = 10000  # You can adjust this based on your vocabulary size
maxlen = 200  # adjust this based on desired sequence length

if os.path.exists('preprocessed_data.pickle'):
    with open('preprocessed_data.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
    X, y, tokenizer, maxlen = loaded_data
else:
    X, y, tokenizer = preprocess_data(data, max_words, maxlen)
    
    with open('preprocessed_data.pickle', 'wb') as f:
        pickle.dump((X, y, tokenizer, maxlen), f)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if 'best_model.h5' file exists
if os.path.exists('best_model.h5'):
    # Load the pre-trained model
    model = load_model('best_model.h5')
else:
    # Create an empty sequential model
    model = Sequential()
    # Add an embedding layer for processing text data
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=maxlen))
    # Add a type of neural layer called LSTM to process sequences
    model.add(LSTM(128, return_sequences=True))
    # Add another LSTM layer for deeper processing
    model.add(LSTM(64))
    # Add a layer with ReLU activation function for more complex patterns
    model.add(Dense(64, activation='relu'))
    # Add dropout to prevent overfitting (a technique to improve model's generalization)
    model.add(Dropout(0.5))
    # Add the final layer for binary classification (0 or 1) using sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model with settings for training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model using the training data with 5 training cycles (epochs)
    # and update model's parameters using batches of 64 data points
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
    model.save('best_model.h5')

# Modify the classify_text function to accept and classify multiple lines of text
def classify_text(input_text):
    # Clean the input text
    cleaned_text = clean_text(input_text)
    
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    
    # Make predictions for each sequence
    predictions = model.predict(padded_sequences)
    
    # Interpret the predictions for each sequence
    results = []
    for prediction in predictions:
        if prediction < 0.5:
            results.append("The text is likely to be fake.")
        else:
            results.append("The text is likely to be real.")
    
    return results

# Collect input
user_input = input("Enter a piece of text to classify: ")
# Classify the input text
results = classify_text(user_input)
# Print the results
for result in results:
    print(result)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)  # The value of the loss function on the test data
print("Test Accuracy:", accuracy)  # The accuracy of the model's predictions on the test data

# Function to create a neural network model with customizable hyperparameters for tuning
def create_model(lstm_units=128, dense_units=64, dropout_rate=0.5, embedding_output=128):
    # Create an empty neural network model
    model = Sequential()
    # Add a layer to process text data (Embedding layer)
    model.add(Embedding(input_dim=max_words, output_dim=embedding_output, input_length=maxlen))
    # Add a layer for sequence processing (LSTM)
    model.add(LSTM(lstm_units, return_sequences=True))
    # Add another LSTM layer with fewer units
    model.add(LSTM(int(lstm_units/2)))
    # Add a layer to learn complex patterns (Dense with ReLU activation)
    model.add(Dense(dense_units, activation='relu'))
    # Add dropout to prevent overfitting
    model.add(Dropout(dropout_rate))
    # Add the final layer for binary classification (Dense with sigmoid activation)
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model for training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Return the created model
    return model

# Define an early stopping callback to monitor validation loss and stop training 
# if it doesn't improve for 'patience' epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# Define a model checkpoint callback to save the best model weights based on validation loss
# The 'save_best_only=True' option ensures that only the best model is saved
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model with specified hyperparameters, and use callbacks for early stopping and checkpointing
model = create_model(lstm_units=256, dense_units=128, dropout_rate=0.3, embedding_output=256)
# Train the model using the training data for 10 epochs:
# - X_train: Input sequences for training
# - y_train: Labels for training
# - epochs: Number of training cycles
# - batch_size: Number of data points processed in each batch
# - validation_split: 20% of the training data used for validation
# - callbacks: EarlyStopping and ModelCheckpoint callbacks used during training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Load the weights of the best performing model
model.load_weights('best_model.h5')

# Evaluate the model's performance on the test dataset
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)