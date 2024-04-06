import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('patientlungcancer.csv')

df['Cancer_Risk'] = df['Level'].apply(lambda x: 1 if x in ['High', 'Medium'] else 0)
# Create synthetic noisy features
np.random.seed(0)
df['Synthetic_Noise1'] = np.random.normal(0, 1, df.shape[0])
df['Synthetic_Noise2'] = np.random.normal(0, 1, df.shape[0])

X = df.drop(['Age', 'Level', 'Cancer_Risk'], axis=1)
y = df['Cancer_Risk']

# Identifying categorical and numerical columns in X, including synthetic features
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64', 'float']]

# Preprocessors
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# RandomForestClassifier with regularization parameters
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict and evaluate the model
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Adjusted Model Accuracy: {accuracy}")
