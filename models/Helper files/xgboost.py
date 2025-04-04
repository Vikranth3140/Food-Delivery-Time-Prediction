# -*- coding: utf-8 -*-
"""XGBoost.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H1Zwlb1hEjWpch2e_MYjlczQXTmc5kMA
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("/content/train.csv")

# Define numerical and categorical features
numerical_features = df[
    [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Vehicle_condition",
        "multiple_deliveries",
    ]
]

categorical_features = df[
    [
        "Weatherconditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
    ]
]

target = df["Time_taken(min)"]
# Encoding using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(categorical_features)
features = pd.concat(
    [pd.DataFrame(encoded_categorical), numerical_features.reset_index(drop=True)],
    axis=1,
)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)  # splitting
# Train the XGBoost model
model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=500, max_depth=7, learning_rate=0.1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
tolerance = 5  # Define a tolerance level
accuracy = (abs(y_pred - y_test) <= tolerance).mean() * 100
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy Measure (within {tolerance} min): {accuracy:.2f}%")
with open("xgboost_metrics.txt", "w") as f:
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"Accuracy-like Measure (within {tolerance} min): {accuracy:.2f}%\n")
