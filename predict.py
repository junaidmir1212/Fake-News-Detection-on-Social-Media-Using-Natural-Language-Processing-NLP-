import pickle
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# Load Logistic Regression model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))

def clean_text(text):
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', '', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def predict_news(text):
    cleaned    = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0]
    label      = 'REAL NEWS' if prediction == 1 else 'FAKE NEWS'
    score      = round(max(confidence) * 100, 2)
    print(f'\nResult    : {label}')
    print(f'Confidence: {score}%')
    print('-' * 35)

sample_texts = [
    'Scientists discover new vaccine that cures all diseases overnight',
    'NASA confirms water found on Mars surface in new study',
    'Donald Trump arrested by FBI for secret meeting with aliens',
    'WHO releases new guidelines on COVID-19 vaccine safety'
]

print('=' * 35)
print('   FAKE NEWS DETECTOR - TEST')
print('=' * 35)

for t in sample_texts:
    print(f'\nText: {t[:60]}...')
    predict_news(t)