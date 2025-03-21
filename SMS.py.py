# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NZJ8TmyK661hJ2ZDeL5_6G5BClN8ctis
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('/spam.csv', encoding='latin-1')

print('Dataset Sample:')
print(df.head())
print(f'Columns: {df.columns}')

df = df.iloc[:, :2]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

print(f'Missing values: {df.isnull().sum().sum()}')

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)

svm = SVC()
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

def evaluate_model(name, y_true, y_pred):
    print(f'--- {name} Model Evaluation ---')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    print('Confusion Matrix:', confusion_matrix(y_true, y_pred))

evaluate_model('Naive Bayes', y_test, nb_pred)
evaluate_model('Logistic Regression', y_test, lr_pred)
evaluate_model('Support Vector Machine', y_test, svm_pred)