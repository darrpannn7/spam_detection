import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load emails from specified folder
def load_emails_from_folder(folder):
    emails = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                emails.append(file.read())
                labels.append(1 if 'spm' in filename.lower() else 0)
    return emails, labels

# Load training and test datasets
train_emails, train_labels = load_emails_from_folder('/path/to/train')
test_emails, test_labels = load_emails_from_folder('/path/to/test')

# Convert emails to a matrix of token counts
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_emails)
X_test = vectorizer.transform(test_emails)

# Train the model (using Random Forest here as an example)
model = RandomForestClassifier()
model.fit(X_train, train_labels)

# Save the model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate and print the accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy: {accuracy:.4f}')
