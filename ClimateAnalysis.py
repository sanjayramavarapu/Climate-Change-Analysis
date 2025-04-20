import pandas as pd
embeddings = pd.read_csv('climate_2_dataset.csv')
print(embeddings.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Identify columns with non-numeric data
non_numeric_cols = embeddings.select_dtypes(include=['object']).columns

# Create a LabelEncoder for each non-numeric column
encoders = {}
for col in non_numeric_cols:
    encoders[col] = LabelEncoder()
    embeddings[col] = encoders[col].fit_transform(embeddings[col])

# Separate features and target variable
X = embeddings.drop('T2M', axis=1)  # Adjust based on your target
y = embeddings['T2M']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R²: {r2}')
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Temperature Anomaly')
plt.ylabel('Predicted Temperature Anomaly')
plt.title('Actual vs Predicted Temperature Anomaly')
plt.show()