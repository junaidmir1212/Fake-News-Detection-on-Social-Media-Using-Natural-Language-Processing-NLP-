import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    text  = text.lower()
    # Keep only letters and spaces
    text  = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text  = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Load merged dataset
df = pd.read_csv('data/news_dataset.csv')

print('Cleaning text data...')

# Apply cleaning to text column
df['clean_text'] = df['text'].apply(clean_text)

# Preview before and after
print('\nOriginal sample:')
print(df['text'].iloc[0][:200])
print('\nCleaned sample:')
print(df['clean_text'].iloc[0][:200])

# Save cleaned dataset
df[['clean_text', 'label']].to_csv('data/clean_dataset.csv', index=False)
print('\nClean dataset saved.')
print('Total rows:', len(df))