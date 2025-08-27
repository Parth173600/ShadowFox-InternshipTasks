import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math



file_path = r"C:\Users\Lenovo\OneDrive\Desktop\ShadowFox\Beginner_BostonHousePrice\HousingData.csv"
boston = pd.read_csv(file_path)

print(boston.head())
boston = boston.dropna()

# Features and target
X = boston.drop("MEDV", axis=1).astype(float)   # all columns except target
y = boston["MEDV"].astype(float)                # target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f} \n R²: {r2:.2f}")

# Visualization
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.show()
