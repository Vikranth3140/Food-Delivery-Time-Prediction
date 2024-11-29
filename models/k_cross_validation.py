import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(categorical_features)
features = pd.concat(
    [pd.DataFrame(encoded_categorical), numerical_features.reset_index(drop=True)],
    axis=1,
)

features.columns = features.columns.astype(str)

n_clusters = 5

kf = KFold(n_splits=5, shuffle=True, random_state=42)
silhouette_scores = []

for train_index, test_index in kf.split(features):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    cluster_labels = kmeans.predict(X_test)

    score = silhouette_score(X_test, cluster_labels)
    silhouette_scores.append(score)

average_score = np.mean(silhouette_scores)

print(f"Average Silhouette Score from K-Fold Cross-Validation: {average_score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(silhouette_scores) + 1), silhouette_scores, marker="o")
plt.title("Silhouette Scores for K-Means across Folds")
plt.xlabel("Fold Number")
plt.ylabel("Silhouette Score")
plt.xticks(range(1, len(silhouette_scores) + 1))
plt.grid()
plt.show()
