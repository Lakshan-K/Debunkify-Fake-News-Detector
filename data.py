import pandas as pd
import numpy as np 
  
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import regularizers 
  
import pprint 
import tensorflow.compat.v1 as tf 
from tensorflow.python.framework import ops 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 
tf.disable_eager_execution() 


fake_csv_path = "news_datasets/Fake.csv"
true_csv_path = "news_datasets/True.csv"

# Read the CSV files
fake_data = pd.read_csv(fake_csv_path)
true_data = pd.read_csv(true_csv_path)
fake_data.head() 
true_data.head()

# Merge datasets
data_merge = pd.concat([fake_data, true_data], axis = 0) 
data_merge.head(10)

# Drop columns
data = data_merge.drop(['title', 'subject', 'date'], axis = 1)