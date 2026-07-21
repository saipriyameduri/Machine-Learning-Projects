"""
Student Score Prediction
--------------------------
Predicts a student's exam score based on the number of hours studied,
using Simple Linear Regression.

Dataset: student_scores.csv (25 real student records: Hours vs Scores)
Source: widely-used public dataset from the "Prediction using Supervised ML" task
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("student_scores.csv")
print(f"Total records: {len(df)}")
print(df.describe(), "\n")

X = df[["Hours"]]
y = df["Scores"]

# ---------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 3. Train Linear Regression model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Learned equation: Score = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}\n")

# ---------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("=== Evaluation Metrics (on held-out test set) ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R^2  : {r2:.4f}\n")

print("Predicted vs Actual (test set):")
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.round(1)})
print(comparison, "\n")

# ---------------------------------------------------------
# 5. Save model for reuse
# ---------------------------------------------------------
joblib.dump(model, "student_score_model.pkl")
print("Saved model -> student_score_model.pkl")

# ---------------------------------------------------------
# 6. Predict for a new value
# ---------------------------------------------------------
hours = 9.25
predicted_score = model.predict([[hours]])[0]
print(f"\nPredicted score for {hours} hours of study/day: {predicted_score:.2f}")
