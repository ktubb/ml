import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('Salary Data.csv').dropna()
y = df['Salary']
X_multi = df[['Years of Experience', 'Age']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)
print("\nMulti-variable Regression Predictions:\n", y_pred_m)
print("\n--- Multi-variable Regression Evaluation ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_m, y_pred_m):.2f}")
print(f"R^2 Score: {r2_score(y_test_m, y_pred_m):.4f}")
