import pandas as pd

# Load dataset
data = pd.read_csv('sentiment_analysis_simple.csv', encoding='utf-8', encoding_errors='ignore')


# Count words in <text> column and add into <word_count> column
def word_count(text):
    return len(text.split())


# Apply the word_count function to the 'text' column and store it in the 'word_count' column
data['word_count'] = data['text'].apply(word_count)

# Save the updated DataFrame back to a CSV file
data.to_csv('sentiment_analysis.csv', index=False, encoding='utf-8')
