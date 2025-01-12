import pandas as pd
import pickle
import statsmodels.api as sm

# Load path for new data
PATH = "/data/VehicleInsuranceClaims_Mystery.csv"
df = pd.read_csv(PATH)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load list of significant features
with open('sigFeatures2.txt') as f:
    significant_features = [line.strip() for line in f]

# Output file for predictions
output_file = 'claims_predictions.csv'

# Format the data
df['Engin_size'] = df['Engin_size'].str.replace('L', '').astype(float)
df['breakdown_date'] = pd.to_datetime(df['breakdown_date'], errors='coerce').dt.strftime('%m%d%y')
df['repair_date'] = pd.to_datetime(df['repair_date'], errors='coerce').dt.strftime('%m%d%y')
df['breakdown_date'] = df['breakdown_date'].astype(float)
df['repair_date'] = df['repair_date'].astype(float)

df = df.drop(columns=['Maker', 'Model', 'Color'])
df = pd.get_dummies(df, columns=['Bodytype', 'Gearbox', 'Fuel_type', 'issue', 'category_anomaly'], dtype=int)
df = df.fillna(df.median())

# Ensure all features used in training are present in the dataframe
missing_features = set(significant_features) - set(df.columns)
for feature in missing_features:
    df[feature] = 0  # Add missing features with default value (e.g., 0)

# Filter dataframe for significant features only (ensure the same order as training)
filteredDf = df[significant_features]

# Add constant column
filteredDf = sm.add_constant(filteredDf)

# Make predictions using the model
predictions = model.predict(filteredDf)

# Write predictions to the output file
with open(output_file, 'w') as f:
    f.write('Claim\n')
    for pred in predictions:
        f.write(f"{pred:.0f}\n")
