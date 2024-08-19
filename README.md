# Sentiment Analysis Project

This repository contains a sentiment analysis project focused on predicting the sentiment of textual data. The project is built using Python and includes several components for model training, prediction, and evaluation.

## Overview

The project involves three main files:

1. **`user_sentiment_prediction.py`**: Predicts sentiment from user-provided text input using a pre-trained Logistic Regression model.
2. **`sentiment_analysis_model.py`**: Contains the code for training the sentiment analysis model, including data preprocessing, model training, and saving the trained model and vectorizer.
3. **`sentiment_model_evaluation.py`**: Evaluates and visualizes the performance of the trained sentiment analysis model using various metrics and dimensionality reduction techniques.

## Files and Their Usage

### `user_sentiment_prediction.py`

- **Purpose**: Predicts sentiment (positive, neutral, or negative) for user input.
- **Functionality**:
  - Loads the pre-trained model and TF-IDF vectorizer.
  - Prompts the user to input text.
  - Cleans and transforms the input text.
  - Predicts and prints the sentiment.
- **Usage**: Run this script as a standalone program. Ensure `sentiment_model.pkl` and `vectorizer.pkl` are in the same directory.

### `sentiment_analysis_model.py`

- **Purpose**: Trains a Logistic Regression model for sentiment analysis.
- **Functionality**:
  - Loads and preprocesses the dataset.
  - Trains a Logistic Regression model.
  - Optionally performs hyperparameter tuning using GridSearchCV.
  - Saves the trained model and vectorizer.
- **Files Generated**:
  - `sentiment_model.pkl`: Contains the trained model.
  - `vectorizer.pkl`: Contains the TF-IDF vectorizer.

### `sentiment_model_evaluation.py`

- **Purpose**: Evaluates and visualizes the performance of the sentiment analysis model.
- **Functionality**:
  - Loads the trained model and vectorizer.
  - Makes predictions on the test set.
  - Evaluates performance using classification metrics and confusion matrix.
  - Visualizes data in 2D and 3D using PCA.
- **Dependencies**: Requires pre-trained model and vectorizer from `sentiment_analysis_model.py`.

## Installation and Requirements

To run the scripts, ensure you have the following Python packages installed:

- `pandas`
- `scikit-learn`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `umap-learn`

You can install the required packages using pip:

```bash
pip install pandas scikit-learn numpy scipy matplotlib seaborn umap-learn
