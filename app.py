import os
import re
import pickle
import warnings
import numpy as np
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer

nltk.download('stopwords', quiet=True)

app = Flask(__name__, static_folder='image', static_url_path='/image')

warnings.filterwarnings('ignore', category=UserWarning)

# Load Logistic Regression model
lr_model = pickle.load(open('model/model.pkl', 'rb'))
tfidf    = pickle.load(open('model/tfidf.pkl', 'rb'))

# Load LSTM model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

lstm_model = tf.keras.models.load_model('model/lstm_model.keras')
tokenizer  = pickle.load(open('model/tokenizer.pkl', 'rb'))

MAX_LEN    = 200
stop_words = set(stopwords.words('english'))

# LIME explainer — class 0 = Fake, class 1 = Real
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])


def clean_text(text: str) -> str:
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', '', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)


def predict_lr(text: str) -> dict:
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = lr_model.predict(vec)[0]
    try:
        proba      = lr_model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 1)
    except Exception:
        score      = float(lr_model.decision_function(vec)[0])
        confidence = round(min(99.9, 50 + abs(score) * 10), 1)
    label = 'REAL' if pred == 1 else 'FAKE'
    return {'label': label, 'confidence': confidence}


def predict_lstm(text: str) -> dict:
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN)
    score   = float(lstm_model.predict(padded, verbose=0)[0][0])
    label   = 'REAL' if score >= 0.5 else 'FAKE'
    confidence = round(score * 100 if score >= 0.5 else (1 - score) * 100, 1)
    return {'label': label, 'confidence': confidence}


def get_lime_explanation(text: str, num_features: int = 10) -> dict:
    cleaned = clean_text(text)

    if len(cleaned.split()) < 3:
        return {'fake_words': [], 'real_words': []}

    def lr_predict_proba(texts):
        vecs = tfidf.transform([clean_text(t) for t in texts])
        try:
            return lr_model.predict_proba(vecs)
        except Exception:
            scores = lr_model.decision_function(vecs)
            proba  = np.zeros((len(texts), 2))
            for i, s in enumerate(scores):
                p = 1 / (1 + np.exp(-s))
                proba[i] = [1 - p, p]
            return proba

    try:
        exp = explainer.explain_instance(
            cleaned,
            lr_predict_proba,
            num_features=num_features,
            num_samples=500
        )

        fake_words = []
        real_words = []

        for word, weight in exp.as_list():
            entry = {'word': word, 'weight': round(abs(weight) * 100, 1)}
            if weight < 0:
                fake_words.append(entry)
            else:
                real_words.append(entry)

        fake_words.sort(key=lambda x: x['weight'], reverse=True)
        real_words.sort(key=lambda x: x['weight'], reverse=True)

        return {
            'fake_words': fake_words[:5],
            'real_words': real_words[:5]
        }

    except Exception as e:
        return {'fake_words': [], 'real_words': [], 'error': str(e)}


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

    agree       = lr['label'] == lstm['label']
    final_label = lr['label'] if agree else lstm['label']
    avg_conf    = round((lr['confidence'] + lstm['confidence']) / 2, 1)

    explanation = get_lime_explanation(text)

    return jsonify({
        'lr':          lr,
        'lstm':        lstm,
        'final':       {'label': final_label, 'confidence': avg_conf, 'agree': agree},
        'explanation': explanation
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)