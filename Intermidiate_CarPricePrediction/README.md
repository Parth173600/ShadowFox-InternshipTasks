
# 🏎 Car Price Prediction – ShadowFox AIML Internship

## 📌 Project Overview

This project is part of the **Intermediate Level Task** for the **ShadowFox AIML Internship**.
The goal is to develop a **machine learning regression model** that predicts the **selling price of cars** based on features like fuel type, age, kilometers driven, transmission, and seller type.

---

## 🎯 Objectives

* Load and preprocess the dataset.
* Handle missing values and encode categorical variables.
* Train regression models to predict car selling price.
* Evaluate models using RMSE and R² Score.
* Deployable structure (Flask/Streamlit optional, but not required for minimum task).

---

## ⚙ Tech Stack

* **Language:** Python 🐍
* **Libraries:**

  * pandas, numpy → Data handling
  * scikit-learn → ML models & evaluation
  * matplotlib, seaborn → Visualizations

---

## 🔑 Steps Followed

### 1. Data Preprocessing

* Loaded dataset from CSV.
* Removed irrelevant columns (like `Car_Name`).
* Derived `car_age` from `Year`.
* Encoded categorical features (`Fuel_Type`, `Seller_Type`, `Transmission`).
* Split into training (80%) and testing (20%).

### 2. Model Training

* Baseline: **Linear Regression**
* Improved: **Random Forest Regressor**

### 3. Model Evaluation

* RMSE (Root Mean Squared Error)
* R² Score

📌 Example (replace with your results):

* Linear Regression → RMSE: 3.21, R²: 0.82
* Random Forest → RMSE: 2.45, R²: 0.91

---

## 📊 Visualizations

* Distribution of car prices
* Correlation heatmap
* Predicted vs Actual prices

---

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/<your-username>/ShadowFox.git
   cd ShadowFox/Intermediate_CarPricePrediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   python car_price.py
   ```

---

## 🗂 Dataset

The dataset is provided by ShadowFox:
[Car Price Dataset](https://drive.google.com/file/d/1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z/view?usp=drive_link)

---

## 👨‍💻 Author

Parth Dasharathbhai Prajapati
ShadowFox AIML Intern

---