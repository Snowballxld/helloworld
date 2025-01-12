

X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# List to store RMSE values
rmse = []

# KFold cross-validation
kfold = KFold(n_splits=2, shuffle=True)
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

print(model.summary())
avgRMSE = np.mean(rmse)
print('Average RMSE:', avgRMSE)
print("list of RMSE:", rmse)