# car_price.py
# ShadowFox AIML Internship - Intermediate Task: Car Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# 1. Load dataset
# -------------------------------
file_path =r"C:\Users\Lenovo\OneDrive\Desktop\ShadowFox\Intermidiate_CarPricePrediction\car.csv"
  # use your uploaded CSV
df = pd.read_csv(file_path)


print(df.head())

# -------------------------------
# 2. Drop unnecessary columns
# -------------------------------
if "Car_Name" in df.columns:
    df = df.drop("Car_Name", axis=1)

# Convert year to car age (if Year exists)
if "Year" in df.columns:
    df["Car_Age"] = 2025 - df["Year"]
    df = df.drop("Year", axis=1)

# -------------------------------
# 3. Encode categorical features
# -------------------------------
le = LabelEncoder()
for col in ["Fuel_Type", "Seller_Type", "Transmission"]:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# -------------------------------
# 4. Define features and target
# -------------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# -------------------------------
# 5. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 6. Train and Evaluate Models
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n🔹 {name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Visualization
    plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name} - Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.show()
