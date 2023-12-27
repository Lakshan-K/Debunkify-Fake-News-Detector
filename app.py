from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string


app = Flask(__name__)

# Load the model and tokenizer
model = load_model('news_model.h5')
with open('tokenizer.pickle', 'rb') as f:
    _, _, tokenizer, maxlen = pickle.load(f)

def clean_text(text):
    # clean_text function as defined in the script
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) 
    text = re.sub(r'in', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.replace('\n', ' ')
    return text

# Modify the classify_text function to use the clean_text and make predictions
def classify_text(input_text):
    cleaned_text = clean_text(input_text)  # Use the clean_text function
    sequences = tokenizer.texts_to_sequences([cleaned_text])  # Tokenize text
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)  # Pad sequences
    predictions = model.predict(padded_sequences)  # Make predictions
    results = []
    for prediction in predictions:
        if prediction < 0.5:
            results.append("The text is likely to be fake.")
        else:
            results.append("The text is likely to be real.")
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = classify_text(text)
        return render_template('result.html', prediction=result[0])

if __name__ == '__main__':
    app.run(debug=True)