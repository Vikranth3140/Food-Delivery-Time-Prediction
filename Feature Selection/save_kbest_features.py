import pandas as pd

# Load the dataset
df = pd.read_csv("../Datasets/new/train.csv")

# Selected features based on importance
selected_features = [
    "Road_traffic_density",
    "Festival",
    "multiple_deliveries",
    "Delivery_person_Ratings",
    "Delivery_person_Age",
    "City",
    "Weatherconditions",
    "Vehicle_condition",
    "Type_of_vehicle",
]

# Add the target column to the selected features
selected_features.append("Time_taken(min)")

# Keep only the selected features in the dataset
filtered_df = df[selected_features]

# Save the updated dataset to a new CSV file
filtered_df.to_csv("../Datasets/kbest features/kbest_features.csv", index=False)

print(f"Dataset saved with selected features: {selected_features}")
