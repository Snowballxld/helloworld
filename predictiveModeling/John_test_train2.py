import pickle
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Loading path
PATH = "/data/VehicleInsuranceClaims.csv"
df = pd.read_csv(PATH)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Formatting Data
df['Engin_size'] = df['Engin_size'].str.replace('L', '').astype(float)
df['breakdown_date'] = pd.to_datetime(df['breakdown_date'], errors='coerce').dt.strftime('%m%d%y')
df['repair_date'] = pd.to_datetime(df['repair_date'], errors='coerce').dt.strftime('%m%d%y')
df['breakdown_date'] = df['breakdown_date'].astype(float)
df['repair_date'] = df['repair_date'].astype(float)

df = df.drop(columns=['Maker', 'Model', 'Color'])
df = pd.get_dummies(df, columns=['Bodytype', 'Gearbox', 'Fuel_type', 'issue', 'category_anomaly'], dtype=int)
df = df.fillna(df.median())

y = df[['Claim']]
X = df.drop(columns=['Claim']).copy()

# Applying Chi Square Test
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
chi_selector = SelectKBest(score_func=chi2, k=30)
chi_selector.fit(X_scaled, y)

selected_features_chi2 = X.columns[chi_selector.get_support()]
X_scaled_chi2 = X[selected_features_chi2]

# Cross Validation setup
kf = KFold(n_splits=8, shuffle=True)
all_selected_features = []

# Loop over K-Folds
for train_index, test_index in kf.split(X_scaled_chi2):
    X_train, X_test = X_scaled_chi2.iloc[train_index], X_scaled_chi2.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Forward Feature Selection (FFS)
    selected_features = []
    remaining_features = list(selected_features_chi2)
    best_rmse = float('inf')

    # Forward Feature Selection (FFS) loop - ensure consistency of columns
    for _ in range(len(remaining_features)):
        best_feature = None

        for feature in remaining_features:
            temp_features = selected_features + [feature]
            X_train_temp = X_train[temp_features]
            X_test_temp = X_test[temp_features]

            # Add constant when selecting features
            X_train_temp = sm.add_constant(X_train_temp, has_constant='add')
            X_test_temp = sm.add_constant(X_test_temp, has_constant='add')

            model = sm.OLS(y_train, X_train_temp).fit()
            predictions = model.predict(X_test_temp)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            if rmse < best_rmse:
                best_rmse = rmse
                best_feature = feature

        if best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    # Apply Recursive Feature Elimination (RFE)
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=19)
    X_train_ffr = X_train[selected_features]
    X_train_ffr = sm.add_constant(X_train_ffr, has_constant='add')  # Add constant during RFE
    rfe.fit(X_train_ffr, y_train)

    rfe_selected_features = X_train_ffr.columns[rfe.support_].tolist()
    all_selected_features.append(rfe_selected_features)

# Count how often each feature was selected across folds
selected_feature_counts = pd.Series([item for sublist in all_selected_features for item in sublist]).value_counts()
selected_features_final = selected_feature_counts[selected_feature_counts >= 5].index.tolist()
selected_features_final = [feature for feature in selected_features_final if feature != 'const']  # Remove constant feature

# Build final model with selected features
X_final = X[selected_features_final]

# Cross-validation setup for final model
kf = KFold(n_splits=8, shuffle=True)
final_rmse_scores = []

# Loop over K-Folds for final model evaluation
for train_index, test_index in kf.split(X_final):
    X_train, X_test = X_final.iloc[train_index], X_final.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Add constant to both training and test sets
    X_train_final_filtered = sm.add_constant(X_train, has_constant='add')
    X_test_final_filtered = sm.add_constant(X_test, has_constant='add')

    model = sm.OLS(y_train, X_train_final_filtered).fit()

    # Loop to remove features with p-values > 0.05
    while True:
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > 0.05:
            non_significant_feature = p_values.idxmax()
            X_train_final_filtered = X_train_final_filtered.drop(columns=[non_significant_feature])
            X_test_final_filtered = X_test_final_filtered.drop(columns=[non_significant_feature], errors='ignore')
            model = sm.OLS(y_train, X_train_final_filtered).fit()
        else:
            break

    predictions = model.predict(X_test_final_filtered)
    final_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    final_rmse_scores.append(final_rmse)

# Print overall results
print("\nModel Summary:")
print(model.summary())

print(f"\nFinal RMSE: {np.sqrt(np.mean(final_rmse_scores)):.4f}")

# Create list of significant features based on what is remaining in the model
significant_features = X_test_final_filtered.columns

# Remove the const feature
significant_features = [feature for feature in significant_features if feature != 'const']

# Create txt file with list of significant variables
with open('sigFeatures2.txt', 'w') as f:
    for feature in significant_features:
        f.write(f"{feature}\n")

# Save model into pkl file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
