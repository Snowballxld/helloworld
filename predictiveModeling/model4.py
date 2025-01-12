import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

PATH = "/data/grades_V2.csv"
df = pd.read_csv(PATH)

df2 = df.copy()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = df.drop(columns=['school','sex', 'address', 'famsize', 'Pstatus',
                                 'schoolsup', 'famsup', 'paid', 'activities',
                                 'nursery', 'higher', 'internet', 'romantic',
                                 'Mjob', 'Fjob', 'reason', 'guardian'])

# Imputing missing data with mean
df = df.fillna(df.median())

# Getting target and predictor categories
y = df['grade']
X = df.drop(columns=['grade']).copy()

# list to store RMSE values
rmse = []

# KFold cross-validation
# Optional: add random_state for reproducibility
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kfold.split(X):

    # creating train-test splits
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # adding intercept
    X_train = sm.add_constant(X_train)

    # Continuously loop until all insignificant categories are removed
    while True:
        model = sm.OLS(y_train, X_train).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()  # Get the maximum p-value

        # Check if the maximum p-value is above 0.05
        if max_p_value > 0.05:
            # Identify and drop the feature with the highest p-value
            non_significant_feature = p_values.idxmax()
            X_train = X_train.drop(columns=[non_significant_feature])
        else:
            break


    # Get the final list of remaining features
    remaining_features = X_train.columns
    # Filter X_test to match the selected features in X_train
    X_test_filtered = X_test[remaining_features.drop('const', errors='ignore')]  # Drop 'const' if present
    # Add constant to the filtered X_test
    X_test_filtered = sm.add_constant(X_test_filtered, has_constant='add')  # Add constant here

    # Ensure the same order of columns
    X_test_filtered = X_test_filtered[remaining_features]

    # Final prediction using the reduced model
    predictions = model.predict(X_test_filtered)

    # Calculate RMSE and store it
    rmse.append(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Print model summary for the last model fit
print(model.summary())

# Calculate and print average RMSE
avgRMSE = np.mean(rmse)
print('Average RMSE:', avgRMSE)
