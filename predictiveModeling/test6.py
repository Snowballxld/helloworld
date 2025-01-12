import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
PATH = "/data/computerPurchase.csv"
df = pd.read_csv(PATH)

print(df[["Age", "EstimatedSalary"]].describe())

# Separate into x and y values.
X = df[["Age", "EstimatedSalary"]]
y = df['Purchased']

## SECTION A ########################################
# Split data.
from sklearn.preprocessing import RobustScaler
sc_x = RobustScaler()
X_Scale = sc_x.fit_transform(X)
# Split data.
X_train, X_test, y_train, y_test = train_test_split(
X_Scale, y, test_size=0.25, random_state=0)
X_train_scaled = sc_x.fit_transform(X_train) # Fit and transform X.
X_test_scaled = sc_x.transform(X_test) # Transform X.
## SECTION A ########################################

## SECTION B ########################################
# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True,solver='liblinear')
# Fit the model.
logisticModel.fit(X_train_scaled, y_train)
y_pred = logisticModel.predict(X_test_scaled)
## SECTION B ########################################

# Show confusion matrix and accuracy scores.
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(cm)