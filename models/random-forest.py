import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# df = pd.read_csv("../Datasets/new/train.csv")
df = pd.read_csv("../Datasets/kbest features/kbest_features.csv")

X = df.drop(columns=["Time_taken(min)"])
y = df["Time_taken(min)"]

categorical_columns = X.select_dtypes(include=["object"]).columns

label_encoder = LabelEncoder()

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

for n_estimators in tqdm(range(100, 1001, 100), desc="Training Random Forest"):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    results.append(
        {
            "n_estimators": n_estimators,
            "r2_score": r2,
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
        }
    )

    print(f"Number of estimators: {n_estimators}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("-" * 50)

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(results_df["n_estimators"], results_df["r2_score"], marker="o")
plt.title("R² Score vs Number of Estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("R² Score")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(
    results_df["n_estimators"],
    results_df["mean_absolute_error"],
    marker="o",
    color="red",
)
plt.title("MAE vs Number of Estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("Mean Absolute Error (MAE)")
plt.grid(True)
plt.show()
