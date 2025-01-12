import pandas as pd
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)

pd.set_option('display.width', 1000)

# Import data into a DataFrame.
path = "/data/titanic_training_data.csv"
df = pd.read_csv(path)

numeric_columns = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']
df_numeric = df[numeric_columns]

imputer = KNNImputer(n_neighbors=5)

dfNumeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_columns)

print(dfNumeric.describe())

print(dfNumeric.head(11))
