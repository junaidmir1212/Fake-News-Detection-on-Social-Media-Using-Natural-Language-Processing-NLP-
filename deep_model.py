import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Use tensorflow.keras only — standalone keras package conflicts with tensorflow 2.x
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

print("TensorFlow version:", tf.__version__)

# Load preprocessed dataset
df = pd.read_csv('data/clean_dataset.csv').dropna()
X = df['clean_text'].values
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# Tokenize text sequences
MAX_WORDS = 20000
MAX_LEN   = 200

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN)

# Build LSTM model
model = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Train
print("\nStarting training...")
model.fit(
    X_train_seq, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Evaluate
y_pred = (model.predict(X_test_seq) > 0.5).astype(int)
print("\nLSTM Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Save model and tokenizer
model.save('model/lstm_model.keras')
pickle.dump(tokenizer, open('model/tokenizer.pkl', 'wb'))
print("LSTM model saved successfully.")