import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

train = pd.read_csv("../Datasets/new/train.csv")
print(train.head())

output_dir = "EDA_plots"  # a Directory to save the plots
os.makedirs(output_dir, exist_ok=True)

palette = "Set3"  # color palette for better visualization

categorical_cols = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y="Time_taken(min)", data=train, palette=palette)
    plt.title(f"Box plot of Time_taken(min) by {col}", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
    plt.close()

sns.pairplot(train)  # Pair Plot
plt.savefig(os.path.join(output_dir, "pairplot.png"))
plt.close()
# Plotting histograms for all numeric features
numeric_features = train.select_dtypes(include=["float64", "int64"]).columns
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(train[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"histogram_{feature}.png"))
    plt.close()

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=col, y="Time_taken(min)", data=train)
    plt.title(f"Violin plot of Time_taken(min) by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"violinplot_{col}.png"))  # Save the figure
    plt.close()
