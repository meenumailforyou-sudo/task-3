import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("sales-of-shampoo-over-a-three-ye.csv")

df.columns = ["Month", "Sales"]

# FIX: force numeric index instead of datetime (IMPORTANT FIX)
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df = df.dropna()

# -------------------------
# Trend plot (FIXED)
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(df["Sales"].values, marker="o")
plt.title("Historical Sales Data")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()

# -------------------------
# Feature engineering
# -------------------------
df["Time"] = np.arange(len(df))

X = df[["Time"]]
y = df["Sales"]

# -------------------------
# Train-test split
# -------------------------
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# -------------------------
# Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Prediction
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()

# -------------------------
# Future prediction
# -------------------------
future = np.arange(len(df), len(df)+6).reshape(-1,1)
future_pred = model.predict(future)

print("\nNext 6 months prediction:")
print(future_pred)