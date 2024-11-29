import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# df = pd.read_csv("../Datasets/new/train.csv")
df = pd.read_csv("../Datasets/kbest features/kbest_features.csv")

X = df.drop(columns=["Time_taken(min)"])
y = df["Time_taken(min)"]

categorical_columns = X.select_dtypes(include=["object"]).columns

label_encoder = LabelEncoder()

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = DecisionTreeRegressor(random_state=42)

model = BaggingRegressor(estimator=base_model, n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_val)), y_val, label="Actual", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6)
plt.title("Bagging: Actual vs Predicted Delivery Time")
plt.xlabel("Samples")
plt.ylabel("Time Taken (min)")
plt.legend()
plt.show()
