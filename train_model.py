import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# Target and features
y = df["salary_in_usd"]
X = df.drop("salary_in_usd", axis=1)

# Make sure all features are numeric
X = X.astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tuned Decision Tree
model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print("MAE:", round(mae, 2))
print("R2:", round(r2, 4))
print("Train R2:", round(train_r2, 4))
print("Test R2:", round(test_r2, 4))

# Save model and feature column order
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("Model saved as model.pkl")
print("Model columns saved as model_columns.pkl")