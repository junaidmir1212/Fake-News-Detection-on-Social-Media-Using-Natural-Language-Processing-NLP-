# Fake News Detection on Social Media Using Natural Language Processing (NLP)
### Setup & Run Guide

---

## Requirements
- Windows 10 or 11
- Python 3.11.9 (download link below)
- Internet connection (for first-time library install only)

---

## Step 1 — Install Python 3.11.9

Download and install from:
https://www.python.org/downloads/release/python-3119/

> **Important:** During installation, tick the checkbox **"Add Python 3.11 to PATH"**

---

## Step 2 — Extract the Project

Extract this zip file to your Desktop or any folder.

---

## Step 3 — Open Terminal in Project Folder

1. Open the extracted `fake-news-detector` folder
2. Hold **Shift** + **Right-click** anywhere inside the folder
3. Select **"Open PowerShell window here"**

---

## Step 4 — Install Libraries (run once)

Paste this command in the terminal and press Enter:

```
py -3.11 -m pip install -r requirements.txt
```

Wait for all packages to download and install (~3-5 minutes).

---

## Step 5 — Run the Application

```
py -3.11 app.py
```

You will see output like:
```
* Running on http://127.0.0.1:5000
```

---

## Step 6 — Open in Browser

Open your browser and go to:

```
http://127.0.0.1:5000
```

The Fake News Detection on Social Media Using Natural Language Processing (NLP) application will load.

---

## How to Use

1. Paste any news article or headline into the text box
2. Click **Analyze News** (or press Ctrl+Enter)
3. View results from both models — Logistic Regression and LSTM
4. Use the sample pills at the bottom to try pre-loaded examples

---

## Project Folder Structure

```
fake-news-detector/
├── app.py                  — Flask web server
├── requirements.txt        — Python dependencies
├── README.md               — This file
├── templates/
│   └── index.html          — Web UI
├── model/
│   ├── model.pkl           — Logistic Regression model
│   ├── tfidf.pkl           — TF-IDF vectorizer
│   ├── lstm_model.keras    — LSTM deep learning model
│   └── tokenizer.pkl       — LSTM tokenizer
├── data/
│   └── clean_dataset.csv   — Preprocessed training data
├── train.py                — LR model training script
├── deep_model.py           — LSTM model training script
├── preprocess.py           — Text preprocessing script
└── load_data.py            — Dataset loading script
```

---

## Technical Details

| Model | Type | Accuracy |
|---|---|---|
| Logistic Regression | Classical ML + TF-IDF | 98.72% |
| LSTM | Deep Learning (TensorFlow) | 98.83% |

**Dataset:** 
Kaggle dataset                       : 44,898 rows


Mendeley total                       : 76,537 rows
  ISOT Fake News Dataset             : 45,386 rows
  Fake News Dataset                  : 20,681 rows
  Fake or Real News Dataset          : 6,332 rows
  Fake News Detection Dataset        : 4,138 rows

Kaggle Dataset Link:

(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Mendeley Dataset Link:                     
(https://data.mendeley.com/datasets/945z9xkc8d/1)

Combined total          : 105,947 rows
Fake news  (label=0)    : 47,453
Real news  (label=1)    : 58,494

**Built with:** Python · Flask · TensorFlow · scikit-learn · NLTK

---

*University of East London — BSc(Hons) Computer Science*

**Copyrights**
© Muhammad Junaid Mir
