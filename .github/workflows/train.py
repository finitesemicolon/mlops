#train.py
# Sample ML Model Training
# This is about Linear Regression script - to train the Model &
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
# Sample datasets
data = {
    "yearsExpereince": [ 1, 2, 3, 4, 5, 6, 7, 8, 9 , 10],
    "Salary": [45000, 50000, 60000, 65000, 70000, 85000, 90000, 105000, 110000, 120000]
}

df = pd.DataFrame(data)
#Split data
X = df[["yearsExpereince"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Model
model = LinearRegression()
model.fit(X_train, y_train)
#predict
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Model Trainined with Mean Squared Error: {mse:.2f}")

# Save the Model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
