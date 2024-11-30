import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("../Datasets/new/train.csv")

# Define numerical and categorical features
numerical_features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
]

categorical_features = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

# Encode categorical variables using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(df[categorical_features])

# Get feature names for the one-hot encoded columns
categorical_feature_names = encoder.get_feature_names_out(categorical_features)

# Combine numerical and one-hot encoded categorical features into a single DataFrame
features = pd.concat(
    [
        pd.DataFrame(encoded_categorical, columns=categorical_feature_names),
        df[numerical_features].reset_index(drop=True),
    ],
    axis=1,
)

# Target variable
target = df["Time_taken(min)"]

# SelectKBest to find important features using f_regression
kbest = SelectKBest(score_func=f_regression, k="all")
kbest.fit(features, target)

# Extract scores for each one-hot encoded feature
feature_scores = pd.DataFrame(
    {
        "Feature": list(categorical_feature_names) + numerical_features,
        "Score": kbest.scores_,
    }
)

# Map scores back to original labels
feature_scores["Original_Label"] = feature_scores["Feature"].apply(
    lambda x: next((label for label in categorical_features if label in x), x)
)

# Aggregate scores by original label
aggregated_scores = (
    feature_scores.groupby("Original_Label")["Score"].sum().reset_index()
)

# Sort by importance
aggregated_scores = aggregated_scores.sort_values(by="Score", ascending=False)

# Print aggregated scores
print(aggregated_scores)

# Save aggregated scores to a file
aggregated_scores.to_csv("aggregated_feature_scores.csv", index=False)
