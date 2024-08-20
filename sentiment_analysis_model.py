import pickle
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.model_selection import cross_val_score
import re
import matplotlib.pyplot as plt

"""
Supervised Machine Learning Model: Sentiment Analysis

This Python file trains a sentiment analysis model using logistic regression, a classification algorithm. 
It includes the following steps:

1. Data Loading: Loads the dataset from a CSV file.
    - The "simple" CSV files have features text | word_count
    - The "complex" CSV files are for higher-dimension input, includes date, time, and platform of text
2. Data Preprocessing: Cleans the text data and prepares features using TF-IDF vectorization.
3. Model Training: Trains a Logistic Regression model on the preprocessed data.
4. Model Optimization: (Optional) GridSearchCV can be used to find the best hyperparameters for the model.
5. Model Saving: Saves the trained model and TF-IDF vectorizer to disk for later use in sentiment prediction.

Usage:
- To use the GridSearchCV functionality to find the best model parameters, uncomment the relevant lines.
- If GridSearchCV is not needed, keep it commented out, and the default model will be trained and saved.

Files Generated:
- 'sentiment_model.pkl': Contains the trained logistic regression model.
- 'vectorizer.pkl': Contains the TF-IDF vectorizer used to transform text data.
"""


# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+', '', text)  # Remove mentions
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


# Parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [100, 200, 500, 1000],
}

# Load dataset
data = pd.read_csv('sentiment_analysis_simple.csv', encoding='utf-8')  # Add <encoding_errors='ignore'> in case of error

# Data Preprocessing
data['cleaned_text'] = data['text'].apply(clean_text)

# Prepare features and target
texts = data['cleaned_text']  # Text data
labels = data['sentiment']  # Sentiment labels

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
y = labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Use GridSearchCV for best model
# grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_
# print("Best parameters found:", grid_search.best_params_)  # Best parameters

# Save trained model and text vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer as well
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
