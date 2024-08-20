import pickle
import pandas as pd
import seaborn as sns
import numpy as np
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
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D


"""
Sentiment Analysis Model Evaluation and Visualization

This script is designed to evaluate a sentiment analysis model that has been pre-trained and saved using pickle. 
The script loads the model and vectorizer, makes predictions on the test set, and evaluates the model's performance 
using various metrics such as classification report, Jaccard index, and confusion matrix.

Additionally, the script provides visualizations of the text data in lower-dimensional space using PCA. It allows for
both 2D and 3D visualizations of the PCA components to observe the separation (or lack thereof) of different sentiment classes.
    - NOTE: The PCA visualization needs some tuning due to high-dimensionality of dataset after text-vectorization.

Key functionalities include:
- Loading the pre-trained sentiment analysis model and its vectorizer
- Making predictions on the test set and evaluating the model using classification metrics
- Visualizing the text data in 2D and 3D using PCA for dimensionality reduction

Packages used:
- pickle: For loading the saved model and vectorizer
- pandas, numpy: For data manipulation and numerical computations
- sklearn: For model evaluation and dimensionality reduction (PCA)
- matplotlib, seaborn: For data visualization
"""



from sentiment_analysis_model import X, X_test, y, y_test


# Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Make predictions on the test set using original model or best model
# y_pred = best_model.predict(X_test)
y_pred = model.predict(X_test)

# Cross-Validation
# scores = cross_val_score(model, X, y, cv=5)
# print(f"Cross-validation scores: {scores}")

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Jaccard Index
jaccard = jaccard_score(y_test, y_pred, average='macro')
print(f'Jaccard Index (Macro Average): {jaccard:.2f}')

# To get Jaccard index for each class
jaccard_per_class = jaccard_score(y_test, y_pred, average=None)
print(f'Jaccard Index per Class: {jaccard_per_class}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
print("\nConfusion Matrix:")
print(cm)

# Visualization using PCA
# Number of points to sample
num_samples = 250

# Randomly sample indices
sample_indices = np.random.choice(X.shape[0], size=num_samples, replace=False)

# Subset the data
X_sampled = X[sample_indices]
y_sampled = y.iloc[sample_indices]

X_sampled_log = np.log1p(X_sampled.toarray())

# Reduce dimensionality of the entire feature set for visualization purposes
# Use X_sorted/y_sorted for cluster mapping OR X/y for standard mapping
# sorted_i = np.argsort(y_sampled)
#
# X_sorted = X_sampled[sorted_i]
# y_sorted = y_sampled[sorted_i]
pca = PCA(n_components=2)

# X_scaled = scale(X_sampled, with_mean=False)  # Change to X_sorted if sorting
X_reduced = pca.fit_transform(X_sampled_log)  # Change to X_scaled if scaling

# Map sentiment labels to numeric values for visualization
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
color_labels = y_sampled.map(sentiment_map)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color_labels, cmap='viridis', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Text Data')
plt.colorbar(scatter, label='Sentiment')
plt.show()


# # Uncomment for 3D plot PCA
# pca = PCA(n_components=3)
# X_reduced_sampled = pca.fit_transform(X_sampled.toarray())
#
# # 3D Plot
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X_reduced_sampled[:, 0], X_reduced_sampled[:, 1], X_reduced_sampled[:, 2], c=color_labels, cmap='viridis', alpha=0.5)
#
# ax.set_xlabel('PCA Component 1')
# ax.set_ylabel('PCA Component 2')
# ax.set_zlabel('PCA Component 3')
# plt.title(f'3D PCA of Text Data (Sampled {num_samples} Points)')
# plt.colorbar(scatter, label='Sentiment')
# plt.show()
