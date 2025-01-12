import pandas as pd
import numpy as np

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

path = "/data/employee_turnover.csv"
df = pd.read_csv(path)
print(df)

# Separate into x and y values.
predictorVariables = list(df.keys())
predictorVariables.remove('turnover')
print(predictorVariables)

# Create X and y values.
X = df[predictorVariables]
y = df['turnover']

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# You imported the libraries to run the experiments. Now, let's see it in action.

# Show chi-square scores for each feature.
# There is 1 degree freedom since 1 predictor during feature evaluation.
# Generally, >=3.8 is good)
test = SelectKBest(score_func=chi2, k=15)
chiScores = test.fit(X, y) # Summarize scores
np.set_printoptions(precision=3)
print("\nPredictor variables: " + str(predictorVariables))
print("Predictor Chi-Square Scores: " + str(chiScores.scores_))

# Another technique for showing the most statistically
# significant variables involves the get_support() function.
# cols = chiScores.get_support(indices=True)
# print(cols)
# features = X.columns[cols]
# print(np.array(features))
#
# for i in range(15):
#     if chiScores.scores_[i] > 3.8:
#         print(str(predictorVariables[i]))
#

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Re-assign X with significant columns only after chi-square test.
X = df[['experience', 'age', 'way', 'industry']]

# Split data.
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,random_state=0)

# Build logistic regression model and make predictions.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',random_state=0)
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)
print(y_pred)
print(y_test)

ZeroAndZero = 0
ZeroAndOne = 0
OneAndZero = 0
OneAndOne = 0

for x in range(len(y_test)):
    # print(y_test.iloc[x])
    if y_test.iloc[x] == 1 and y_pred[x] == 1:
        OneAndOne+=1
    elif y_test.iloc[x] == 0 and y_pred[x] == 0:
        ZeroAndZero+=1
    elif y_test.iloc[x] == 1 and y_pred[x] == 0:
        OneAndZero+=1
    elif y_test.iloc[x] == 0 and y_pred[x] == 1:
        ZeroAndOne+=1

print(str(ZeroAndZero) + " " + str(ZeroAndOne) + " " + str(OneAndZero) + " " + str(OneAndOne))

print((OneAndOne + ZeroAndZero)/(ZeroAndOne + ZeroAndZero + OneAndZero + OneAndOne))

TN = ZeroAndZero # = 3 True Negative (Col 0, Row 0)
FN = ZeroAndOne # = 0 False Negative (Col 0, Row 1)
FP = OneAndZero # = 2 False Positive (Col 1, Row 0)
TP = OneAndOne # = 5 True Positive (Col 1, Row 1)

# TN = 32
# FN = 10
# FP = 3
# TP = 15

print("")
print("True Negative: " + str(TN))
print("False Negative: " + str(FN))
print("False Positive: " + str(FP))
print("True Positive: " + str(TP))

precision = (TP/(FP + TP))
print("\nPrecision: " + str(round(precision, 3)))

recall = (TP/(TP + FN))
print("Recall: " + str(round(recall,3)))

F1 = 2*((precision*recall)/(precision+recall))
print("F1: " + str(round(F1,3)))