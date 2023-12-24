import pandas as pd
import numpy as np 
import re
import string
import tensorflow as tf 
import pprint 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.utils import to_categorical 
from keras import regularizers 
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.python.framework import ops 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 
from keras.callbacks import EarlyStopping, ModelCheckpoint #import for Hyperparameter Tuning: EarlyStopping and ModelCheckpoint

# Define the text cleaning function 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'in', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


fake_csv_path = "news_datasets/Fake.csv"
true_csv_path = "news_datasets/True.csv"

# Read the CSV files
fake_data = pd.read_csv(fake_csv_path)
true_data = pd.read_csv(true_csv_path)
fake_data.head() 
true_data.head()

# Assign labels to dataset
fake_data["label"] = 0 
true_data['label'] = 1

# Merge datasets
data_merge = pd.concat([fake_data, true_data], axis = 0) 
data_merge.head(10)

# Drop columns
data = data_merge.drop(['title', 'subject', 'date'], axis = 1)

# Clean the text data
data['text'] = data['text'].apply(clean_text)

# Text Tokenization
max_words = 10000  # You can adjust this based on your vocabulary size
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['text'])

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(data['text'])

# Padding sequences (adjust maxlen as needed)
maxlen = 200  # adjust this based on desired sequence length
X = pad_sequences(sequences, maxlen=maxlen)

# Labels
y = data['label'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)  # The value of the loss function on the test data
print("Test Accuracy:", accuracy)  # The accuracy of the model's predictions on the test data

# Function to create a neural network model with customizable hyperparameters for tuning
def create_model(lstm_units=128, dense_units=64, dropout_rate=0.5, embedding_output=128):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_output, input_length=maxlen))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(int(lstm_units/2)))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Callbacks for stopping training early if no improvement and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model with specified hyperparameters, and use callbacks for early stopping and checkpointing
model = create_model(lstm_units=256, dense_units=128, dropout_rate=0.3, embedding_output=256)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Load the weights of the best performing model
model.load_weights('best_model.h5')

# Evaluate the model's performance on the test dataset
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)