from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the model and tokenizer
# Ensure these paths correctly point to where your 'news_model.h5' and 'tokenizer.pickle' are stored.
model = load_model('news_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def classify_text(input_text):
    # Preprocess the text in the same way as in data.py
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=54, padding='post', truncating='post')
    # Predict
    prediction = model.predict(padded_sequences, verbose=0)[0][0]
    return "This news is likely true." if prediction >= 0.5 else "This news is likely false."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = classify_text(text)
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)