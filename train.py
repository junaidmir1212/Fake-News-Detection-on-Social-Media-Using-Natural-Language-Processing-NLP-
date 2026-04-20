import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load preprocessed dataset
df = pd.read_csv('data/clean_dataset.csv')
df = df.dropna(subset=['clean_text'])

X = df['clean_text']
y = df['label']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", len(X_train))
print("Testing  samples:", len(X_test))

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Save model and vectorizer
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(tfidf,  open('model/tfidf.pkl', 'wb'))
print("Model saved to model/model.pkl")
print("Vectorizer saved to model/tfidf.pkl")