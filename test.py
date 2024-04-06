from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

df = pd.read_csv('patientlungcancer.csv')
df['Cancer_Risk'] = df['Level'].apply(lambda x: 1 if x in ['High', 'Medium'] else 0)
# Assuming df is your DataFrame
X = df.drop(['Age', 'Level', 'Cancer_Risk'], axis=1)  # Feature matrix
y = df['Cancer_Risk']  # Target variable

# Identifying categorical and numerical columns in X
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Preprocessors
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# RandomForestClassifier as the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Fit the model
pipeline.fit(X_train, y_train)

# Feature importance
feature_importances = pipeline.named_steps['model'].feature_importances_

# Print feature importances
print("Feature importances:\n")
for name, importance in zip(X.columns, feature_importances):
    print(f"{name}: {importance}")
