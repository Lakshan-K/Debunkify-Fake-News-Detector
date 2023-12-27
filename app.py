from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn import preprocessing
import os

# Initialize the Flask application
app = Flask(__name__)

# Load and preprocess data, train model
def train_model():
    # Load data
    data = pd.read_csv("news.csv")  # Adjust the path as necessary
    data = data.drop(["Unnamed: 0"], axis=1)  # Adjust based on your data structure

    # Encode labels
    le = preprocessing.LabelEncoder()
    data['label'] = le.fit_transform(data['label'])

    # Define parameters
    embedding_dim = 50
    max_length = 54
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 3000
    test_portion = 0.1

    # Prepare data
    title = data['title'].tolist()[:training_size]  # Adjust according to your data column
    labels = data['label'].tolist()[:training_size]

    # Tokenizer setup
    tokenizer1 = Tokenizer(oov_token=oov_tok)
    tokenizer1.fit_on_texts(title)
    sequences = tokenizer1.texts_to_sequences(title)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Split data
    split = int(test_portion * training_size)
    training_padded = padded[split:]
    training_labels = np.array(labels[split:])

    # Create the model
    model = Sequential([
        Embedding(len(tokenizer1.word_index) + 1, embedding_dim, input_length=max_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    num_epochs = 10  # Adjust as necessary
    model.fit(training_padded, training_labels, epochs=num_epochs, verbose=2)
    
    return model, tokenizer1

# Load model and tokenizer
model, tokenizer = train_model()

# Route for the home page
@app.route('/')
def index():
    # Replace 'index.html' with the actual name of your homepage template
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting text from the form submission
    text = request.form['text']  # Corrected to use 'text' to match the HTML form
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=54, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]

    # Determine the result
    if prediction >= 0.5:
        result = "This news is likely true."
    else:
        result = "This news is likely false."

    # Send result to the result.html template
    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
