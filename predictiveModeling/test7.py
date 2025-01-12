import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.api as sm
import numpy as np

def showXandYplot(x, y, xtitle, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color='blue')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel('y')
    plt.show()

def showResidualPlotAndRMSE(x, y, predictions):
    xmax = max(x)
    xmin = min(x)
    residuals = y - predictions
    plt.figure(figsize=(8, 3))
    plt.title("Residuals")
    plt.plot([xmin, xmax], [0, 0], '--', color='black')
    plt.scatter(x, residuals, color='red')
    plt.show()
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))

x = [0.01, 0.2, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y = [0.99, 0.819, 0.6065, 0.4966, 0.40657, 0.368, 0.1353, 0.0498,
     0.01831, 0.00674, 0.0025, 4.5399e-05, 1.670e-05, 6.1e-06,
     2.260e-06, 8.3153e-07, 3.0590e-07, 1.12e-07, 4.14e-08,
     1.52e-08, 5.60e-09, 2.061e-09]

print(y)
showXandYplot(x, y, 'x', 'x and y')

dfX = pd.DataFrame({"x": x})
dfY = pd.DataFrame({"y": y})
dfX = sm.add_constant(dfX)

model = sm.OLS(y, dfX).fit()
predictions = model.predict(dfX)

print(model.summary())
showResidualPlotAndRMSE(x, y, predictions)

x_transformed = np.exp(-np.array(x))

showXandYplot(x_transformed, y, 'Transformed x (exp(-x))', 'Transformed x and y')

dfX_transformed = pd.DataFrame({"x_transformed": x_transformed})
dfX_transformed = sm.add_constant(dfX_transformed)

model_transformed = sm.OLS(y, dfX_transformed).fit()
predictions_transformed = model_transformed.predict(dfX_transformed)

print(model_transformed.summary())
showResidualPlotAndRMSE(x_transformed, y, predictions_transformed)
