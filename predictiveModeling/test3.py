# Load libraries
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn import datasets

from sklearn.model_selection import train_test_split

import pandas as pd

# Load data from sklearn library.

iris = datasets.load_iris()

X = iris.data

y = iris.target

# Split data.

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=0)

# Create one-vs-rest logistic regression object

clf = LogisticRegression(

    random_state=0,

    multi_class='multinomial', solver='newton-cg')

# Train model

model = clf.fit(X_train, y_train)

# Predict class

y_pred = model.predict(X_test)

print(y_pred)

# View predicted probabilities

y_prob = model.predict_proba(X_test)

print(y_prob)

# sum_probabilities = np.sum(y_prob, axis=1)
#
# print(sum_probabilities)

# Show precision, recall and F1 scores for all classes.

from sklearn import metrics

precision = metrics.precision_score(y_test, y_pred, average=None)

recall = metrics.recall_score(y_test, y_pred, average=None)

f1 = metrics.f1_score(y_test, y_pred, average=None)

print("Precision: " + str(precision))

print("Recall: " + str(recall))

print("F1: " + str(f1))