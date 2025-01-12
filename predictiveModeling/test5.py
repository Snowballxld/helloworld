from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

import pandas as pd

import numpy as np

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/data/milk.csv")

# Split the data at the start to hold back a test set.

train, test = train_test_split(df, test_size=0.2)

X_train = train.copy()

X_test = test.copy()

del X_train['labels']

del X_test['labels']

del X_train['dates']

del X_test['dates']

y_train = train['labels']

y_test = test['labels']

# Scale X values.

xscaler = StandardScaler()

Xtrain_scaled = xscaler.fit_transform(X_train)

Xtest_scaled = xscaler.transform(X_test)

# Generate PCA components.

pca = PCA(0.8)

# Always fit PCA with train data. Then transform the train data.

X_reduced_train = pca.fit_transform(Xtrain_scaled)

# Transform test data with PCA

X_reduced_test = pca.transform(Xtest_scaled)

print("\nPrincipal Components")

print(pca.components_)

print("\nExplained variance: ")

print(pca.explained_variance_)

# Train regression model on training data

model = LogisticRegression(solver='liblinear')

model.fit(X_reduced_train, y_train)

# Predict with test data.

preds = model.predict(X_reduced_test)

report = classification_report(y_test, preds)
print("------------------")
print(report)

print("------------------")


import matplotlib.pyplot as plt

# Scree plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-', label='Scree')
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.show()

# Cumulative variance plot
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', label='Cumulative Variance')
plt.title("Cumulative Variance Explained")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance")
plt.legend()
plt.show()
