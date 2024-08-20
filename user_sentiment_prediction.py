import pandas as pd
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from sentiment_analysis_model import clean_text

"""
Sentiment Analysis Prediction from User Input

This Python file is designed for predicting the sentiment (positive/neutral/negative) of user-provided text input 
using a pre-trained Logistic Regression model. The script performs the following steps:

1. Load Model and Vectorizer: The pre-trained logistic regression model and TF-IDF vectorizer are loaded from 
   the saved pickle files ('sentiment_model.pkl' and 'vectorizer.pkl').
2. Text Input: The user is prompted to input text via the console.
3. Text Preprocessing: The input text is cleaned using the `clean_text` function from the 
   'sentiment_analysis_model' module to match the preprocessing used during training.
4. Sentiment Prediction: The cleaned text is transformed using the TF-IDF vectorizer, and the sentiment 
   is predicted using the loaded model.
5. Output: The predicted sentiment label (positive/neutral/negative) is printed to the console.

Usage:
- Run this script as a standalone program. When executed, it will prompt the user to enter a piece of text 
  and will output the predicted sentiment based on the trained model.

Note:
- Ensure that the 'sentiment_model.pkl' and 'vectorizer.pkl' files are present in the same directory as this script.
- Run 'sentiment_analysis_model.py' if changes made to model.
"""


# Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# User Text Input Function
def predict_sentiment():
    user_text = str(input("Enter some text: "))
    cleaned_text = clean_text(user_text)

    # Transform text to match model's training data
    X_input = vectorizer.transform([cleaned_text])

    # Predict <user_text>  sentiment
    prediction = model.predict(X_input)

    # Output result
    print(f"The sentiment of the entered text is: {prediction[0]}")


if __name__ == "__main__":
    predict_sentiment()
