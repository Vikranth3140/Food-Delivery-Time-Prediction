import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("../datasets/new/train.csv")

# Define numerical and categorical features based on the cleaned dataset
numerical_features = df[
    ["Delivery_person_Age", "Delivery_person_Ratings", "Restaurant_latitude", 
     "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude", 
     "Vehicle_condition", "multiple_deliveries"]
]

categorical_features = df[
    ["Weatherconditions", "Road_traffic_density", "Type_of_order", 
     "Type_of_vehicle", "Festival", "City"]
]

# Target variable
target = df["Time_taken(min)"]

# Encode categorical variables using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(categorical_features)

# Get feature names for the one-hot encoded columns
categorical_feature_names = encoder.get_feature_names_out(categorical_features.columns)

# Combine numerical and one-hot encoded categorical features into a single DataFrame
features = pd.concat([pd.DataFrame(encoded_categorical), numerical_features.reset_index(drop=True)], axis=1)

# Assign appropriate column names
feature_names = list(categorical_feature_names) + list(numerical_features.columns)
features.columns = feature_names

# SelectKBest to find important features using f_regression
kbest = SelectKBest(score_func=f_regression, k="all")
kbest.fit(features, target)

# Get scores and create a DataFrame for better visualization
feature_scores = pd.DataFrame({"Feature": feature_names, "Score": kbest.scores_})

# Sort features by importance
feature_scores = feature_scores.sort_values(by="Score", ascending=False)

# Print feature scores
print(feature_scores)

# Save feature scores to a file
feature_scores.to_csv("feature_scores.txt", index=False)