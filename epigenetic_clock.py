import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

df = pd.read_csv('patientlungcancer.csv')

X = df.drop(['Age', 'Patient Id'], axis=1) 
y = df['Age']

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object" and cname != 'Level']
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64'] and cname not in ['Patient Id', 'Level']]

# numerical and categorical data
numerical_transformer = StandardScaler()
#print(numerical_transformer)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#print(categorical_transformer)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# model for predicting Age
age_model = RandomForestRegressor(n_estimators=100, random_state=0)

age_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', age_model)])

# fitting the model
age_pipeline.fit(X, y)

# adding predicted Age to the dataset
df['PredictedAge'] = age_pipeline.predict(X)

df.to_csv('updated_patientlungcancer.csv', index=False)

predicted_ages = df['PredictedAge']
actual_ages = y

r2 = r2_score(actual_ages, predicted_ages)

print(f'R^2 score: {r2}')


