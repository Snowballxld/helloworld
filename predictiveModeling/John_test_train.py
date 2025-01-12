import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import KFold
import pickle

# Set Path Here
PATH = "/data/grades_V2.csv"
df = pd.read_csv(PATH)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Using IQR to identify outliers in every categoryx
for column in df.select_dtypes(include=['int64']).columns:

    # Making sure not to 'treat' the target variable
    if column == 'grade':
        continue

    # Getting the quantiles for the outlier identification
    Q1 = df[column].quantile(0.3)
    Q3 = df[column].quantile(0.7)

    # calculating IQR value
    IQR = Q3 - Q1

    # Creating lower and upper bound values
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Replacing outliers with median of column
    df[column] = df[column].where((df[column] >= lower) & (df[column] <= upper), df[column].median())

# Binning categories
df['age'] = pd.cut(df['age'], bins=[0, 15, 17, 24], labels=['teen', 'youth', 'adult'])
df['absences'] = pd.cut(df['absences'], bins=[0, 5, 16, 30], labels=['sometimes', 'often', 'a lot'])
df['health'] = pd.cut(df['health'], bins=[0, 2, 6], labels=['poor', 'good'])

# Generate dummy variables for categorical features
df = pd.get_dummies(df, columns=['school', 'sex', 'address', 'famsize', 'Pstatus',
                                 'schoolsup', 'famsup', 'paid', 'activities',
                                 'nursery', 'higher', 'internet', 'romantic',
                                 'Mjob', 'Fjob', 'reason', 'guardian', 'age'
                                 , 'absences', 'health'
                                 ], dtype=int)

# Imputing missing data with median
df = df.fillna(df.median())

# Getting target and predictor categories
y = df['grade']
X = df.drop(columns=['grade']).copy()

# Scaling predictor categories
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# List to store RMSE values
rmse = []

# KFold cross-validation
kfold = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kfold.split(X):

    # creating train-test splits
    X_train = X_scaled.iloc[train_index]
    X_test = X_scaled.iloc[test_index]
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

    # Get new list of remaining features as we removed a lot and
    # will be error when predicting if we don't change the remaining features
    remaining_features = X_train.columns
    X_test_filtered = X_test[remaining_features.drop('const', errors='ignore')]
    X_test_filtered = sm.add_constant(X_test_filtered, has_constant='add')
    X_test_filtered = X_test_filtered[remaining_features]
    predictions = model.predict(X_test_filtered)

    # Calculate RMSE and store it
    rmse.append(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Print model summary for the last model fit
print(model.summary())

# Calculate and print average RMSE
avgRMSE = np.mean(rmse)
print('Average RMSE:', avgRMSE)

# Create list of significant features based on what is remaining in the model
significant_features = X_test_filtered.columns
# remove the const feature
significant_features = [feature for feature in significant_features if feature != 'const']

# create txt file with list of significant variables
with open('sigFeatures.txt', 'w') as f:
    for feature in significant_features:
        f.write(f"{feature}\n")

# save model into pkl file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
