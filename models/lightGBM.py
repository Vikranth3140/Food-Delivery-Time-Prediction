import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# df = pd.read_csv("../Datasets/new/train.csv")
df = pd.read_csv("../Datasets/kbest features/kbest_features.csv")

label_encoder = LabelEncoder()

categorical_cols = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_vehicle",
    "Festival",
    "City",
]

for col in tqdm(categorical_cols):
    df[col] = label_encoder.fit_transform(df[col])

non_numeric_columns_after = df.select_dtypes(include=["object"]).columns
print(f"Remaining non-numeric columns after encoding: {non_numeric_columns_after}")

X = df.drop(columns=["Time_taken(min)"])
y = df["Time_taken(min)"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm_model = LGBMRegressor(n_estimators=500, random_state=42)
lgbm_model.fit(X, y)
y_pred = lgbm_model.predict(X_val)

r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

print(f"Number of estimators: {500}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
