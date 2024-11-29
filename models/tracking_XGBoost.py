import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# df = pd.read_csv("../Datasets/new/train.csv")
df = pd.read_csv("../Datasets/kbest features/kbest_features.csv")

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
        "Type_of_vehicle",
        "Festival",
        "City",
    ]
]

y = df["Time_taken(min)"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(categorical_features)
X = pd.concat(
    [pd.DataFrame(encoded_categorical), numerical_features.reset_index(drop=True)],
    axis=1,
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=500, max_depth=7, learning_rate=0.1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

r2 = r2_score(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
tolerance = 5
accuracy = (abs(y_pred - y_val) <= tolerance).mean() * 100

print(f"R2 Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Accuracy Measure (within {tolerance} min): {accuracy:.2f}%")
