import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("../../Datasets/new/train.csv")

# Step 1: Prepare features (X) and target (y)
X = df.drop(columns=["Time_taken(min)"])  # Features (all columns except the target)
y = df["Time_taken(min)"]  # Target (Time taken)

# Step 2: Label encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le  # Store the label encoder for future use

# Step 3: Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Initialize the Random Forest model
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Step 5: Train the model
rf.fit(X_train, y_train)

# Step 6: Calculate feature importances
importances = rf.feature_importances_

# Step 7: Get the feature names
features = X.columns

# Step 8: Create a DataFrame for better visualization
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})

# Step 9: Sort the features by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Step 10: Display the sorted feature importances
print(importance_df)

# Step 11: Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()


#  Feature  Importance
# 1       Delivery_person_Ratings    0.225531
# 7          Road_traffic_density    0.149848
# 11          multiple_deliveries    0.127550
# 6             Weatherconditions    0.113034
# 0           Delivery_person_Age    0.087808
# 8             Vehicle_condition    0.071233
# 5   Delivery_location_longitude    0.049383
# 4    Delivery_location_latitude    0.049001
# 3          Restaurant_longitude    0.038088
# 2           Restaurant_latitude    0.034933
# 12                     Festival    0.022154
# 9                 Type_of_order    0.015772
# 13                         City    0.009634
# 10              Type_of_vehicle    0.006032
