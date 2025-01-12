from numpy import dtype
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

# Import Data

path = "/data/carPrice.csv"
df = pd.read_csv(path)

# Enable the display of all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ---------------------------------------------
# Generate quick views of data.
def viewQuickStats():
    print("\n*** Show contents of the file.")
    print(df.head())
    print("\n*** Show the description for all columns.")
    print(df.info())
    print("\n*** Describe numeric values.")
    print(df.describe())
    print("\n*** Showing frequencies.")
    # Show frequencies.
    print(df['model'].value_counts())
    print("")
    print(df['transmission'].value_counts())
    print("")
    print(df['fuel type'].value_counts())
    print("")
    print(df['engine size'].value_counts())
    print("")
    print(df['fuel type2'].value_counts())
    print("")
    print(df['year'].value_counts())
    print("")
# ---------------------------------------------
# Fix the price column.
for i in range(0, len(df)):
    priceStr = str(df.iloc[i]['price'])
    priceStr = priceStr.replace("£", "")
    riceStr = priceStr.replace("-", "")
    priceStr = priceStr.replace(",", "")
    df.at[i, 'price'] = priceStr

# Convert column to number.
df['price'] = pd.to_numeric(df['price'])

# ---------------------------------------------
# Fix the price column.
averageYear = df['year'].mean()
for i in range(0, len(df)):
    year = df.iloc[i]['year']
    if (np.isnan(year)):
        df.at[i, 'year'] = averageYear

# ---------------------------------------------
# Fix the engine size2 column.
for i in range(0, len(df)):
    try:
        engineSize2 = df.loc[i]['engine size2']
        if (pd.isna(engineSize2)):
            df.at[i, 'engine size2'] = "0"
    except Exception as e:
        error = str(e)
        print(error)

df['engine size2'] = pd.to_numeric(df['engine size2'])
df['mileage2'].value_counts()
viewQuickStats()
# ---------------------------------------------
# Fix the mileage column.
for i in range(0, len(df)):
    mileageStr = str(df.iloc[i]['mileage'])
    mileageStr = mileageStr.replace(",", "")
    df.at[i, 'mileage'] = mileageStr
    try:
        if (not mileageStr.isnumeric()):
            df.at[i, 'mileage'] = "0"
    except Exception as e:
        error = str(e)
        print(error)

df['mileage'] = pd.to_numeric(df['mileage'])
viewQuickStats()

# Compute the correlation matrix
corr = df.corr(numeric_only=True)
# plot the heatmap
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)
#plt.show()


df = pd.get_dummies(df, columns=['transmission','fuel type2'], dtype = int)
X = df[['engine size2', 'year', 'transmission_Automatic', 'transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto', 'fuel type2_Diesel', 'fuel type2_Hybrid', 'fuel type2_Other', 'fuel type2_Petrol']].values

df['2020Bin'] = pd.cut(x=df['year'], bins=[2019,2020])
df['2019Bin'] = pd.cut(x=df['year'], bins=[2018,2019])
df['2018Bin'] = pd.cut(x=df['year'], bins=[2017,2018])
df['2017Bin'] = pd.cut(x=df['year'], bins=[2016,2017])
df['2016Bin'] = pd.cut(x=df['year'], bins=[2015,2016])
df['2015Bin'] = pd.cut(x=df['year'], bins=[2014,2015])
df['2014Bin'] = pd.cut(x=df['year'], bins=[2013,2014])
df['2013Bin'] = pd.cut(x=df['year'], bins=[2012,2013])
df['OtherBin'] = pd.cut(x=df['year'], bins=[0,2012])

#df = pd.get_dummies(df, columns=['2020Bin', '2019Bin', '2018Bin', '2017Bin', '2016Bin', '2015Bin', '2014Bin', '2013Bin', 'OtherBin'], dtype=int)

#X = df[['engine size2', 'year', '2020Bin_(2019, 2020]', '2019Bin_(2018, 2019]', '2018Bin_(2017, 2018]', '2017Bin_(2016, 2017]', '2016Bin_(2015, 2016]', '2015Bin_(2014, 2015]', '2014Bin_(2013, 2014]', '2013Bin_(2012, 2013]', 'OtherBin_(0, 2012]']].values

#X = df[['engine size2', 'year']].values

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.

X = sm.add_constant(X)
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())
print()
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(df.head(100))

def getPriceCar(engine_size2, year, transmission_Automatic, transmission_Manual, transmission_Other, transmission_Semi_Auto, fuel_type2_Diesel, fuel_type2_Hybrid, fuel_type2_Other, fuel_type2_Petrol, actualPrice):
    price = -4066000 + (0.6156 * engine_size2) + (2530.7387 * year) - (1016000 * transmission_Automatic) - (1021000 * transmission_Manual) - (1013000 * transmission_Other) - (1016000 * transmission_Semi_Auto) - (569.3070 * fuel_type2_Diesel) - (2234.9442 * fuel_type2_Hybrid) - (1320.2621 * fuel_type2_Other) + (2590.1761 * fuel_type2_Petrol)
    print("Car actual price: " + str(actualPrice) + "   predicted price: " + str(price))

getPriceCar(2.1, 2017, 0, 0, 0, 1, 1, 0, 0, 0, 33000)