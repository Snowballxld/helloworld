import pandas as pd
import pickle
import statsmodels.api as sm

# gets path here
PATH = "/data/grades_V2.csv"
df = pd.read_csv(PATH)

# opens saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# gets list of significant features here
with open('sigFeatures.txt') as f:
    significant_features = [line.strip() for line in f]

# Output file for predictions
output_file = 'grades_predictions.csv'

#Preparing Data
# Binning categories
df['age'] = pd.cut(df['age'], bins=[0, 15, 17, 24], labels=['teen', 'youth', 'adult'])
df['absences'] = pd.cut(df['absences'], bins=[0, 5, 16, 20], labels=['sometimes', 'often', 'a lot'])
df['health'] = pd.cut(df['health'], bins=[0, 2, 6], labels=['poor', 'good'])

# Generate dummy variables for categorical features
df = pd.get_dummies(df, columns=['school', 'sex', 'address', 'famsize', 'Pstatus',
                                 'schoolsup', 'famsup', 'paid', 'activities',
                                 'nursery', 'higher', 'internet', 'romantic',
                                 'Mjob', 'Fjob', 'reason', 'guardian'
                                 , 'age'
                                 , 'absences'
                                 , 'health'
                                 ], dtype=int)

# Imputing missing data with mean
df = df.fillna(df.mean())

# filter df for significant features only
filteredDf = df[significant_features]

# add constant
filteredDf = sm.add_constant(filteredDf)

# make predictions with model
predictions = model.predict(filteredDf)

#write predictions to output file
with open(output_file, 'w') as f:
    f.write('grade\n')
    for pred in predictions:
        f.write(f"{pred:.3f} \n")