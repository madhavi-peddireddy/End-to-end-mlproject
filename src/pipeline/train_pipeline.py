from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Assuming 'data' is your DataFrame and 'target' is the target variable
# Define your preprocessing steps
categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
numeric_features = ['reading_score', 'writing_score']

# Create the preprocessing pipelines for both numeric and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Fit the model
model.fit(data, target)

# Save the model and preprocessor
with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model.named_steps['regressor'], f)
with open('artifacts/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
