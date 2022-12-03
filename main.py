import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the data
df = pd.read_csv("./data/news.csv")

# Gets the shape and head
df.shape
df.head()

# Gets the labels from the data frame
labels = df.label
labels.head()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7)

# Initializing a TFidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggresiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculates accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round((score * 100), 2)}%")

# Building a Confusion Matrix
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
