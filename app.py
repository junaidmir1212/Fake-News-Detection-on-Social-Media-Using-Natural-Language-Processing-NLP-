import os
import re
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)


app = Flask(__name__, static_folder='image', static_url_path='/image')


# ── Load Logistic Regression model ──────────────────────────────────────────
lr_model = pickle.load(open('model/model.pkl', 'rb'))
tfidf    = pickle.load(open('model/tfidf.pkl', 'rb'))

# ── Load LSTM model ──────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

lstm_model = tf.keras.models.load_model('model/lstm_model.keras')
tokenizer  = pickle.load(open('model/tokenizer.pkl', 'rb'))

MAX_LEN    = 200
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', '', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)


def predict_lr(text: str) -> dict:
    cleaned   = clean_text(text)
    vec       = tfidf.transform([cleaned])
    pred      = lr_model.predict(vec)[0]
    proba     = lr_model.predict_proba(vec)[0]
    label     = 'REAL' if pred == 1 else 'FAKE'
    confidence = round(float(max(proba)) * 100, 1)
    return {'label': label, 'confidence': confidence}


def predict_lstm(text: str) -> dict:
    cleaned  = clean_text(text)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(seq, maxlen=MAX_LEN)
    score    = float(lstm_model.predict(padded, verbose=0)[0][0])
    label    = 'REAL' if score >= 0.5 else 'FAKE'
    confidence = round(score * 100 if score >= 0.5 else (1 - score) * 100, 1)
    return {'label': label, 'confidence': confidence}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text or len(text) < 20:
        return jsonify({'error': 'Please enter at least 20 characters of news text.'}), 400

    lr   = predict_lr(text)
    lstm = predict_lstm(text)

    # Ensemble: both models must agree for high confidence verdict
    agree = lr['label'] == lstm['label']
    final_label = lr['label'] if agree else ('REAL' if lr['label'] == 'REAL' else 'FAKE')
    avg_conf    = round((lr['confidence'] + lstm['confidence']) / 2, 1)

    return jsonify({
        'lr':    lr,
        'lstm':  lstm,
        'final': {'label': final_label, 'confidence': avg_conf, 'agree': agree}
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)