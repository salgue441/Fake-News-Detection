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
    df['text'], labels, test_size=2.0, random_state=7)
