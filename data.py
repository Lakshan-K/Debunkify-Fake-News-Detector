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
  
from tensorflow.python.framework import ops 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 
tf.disable_eager_execution() 

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
