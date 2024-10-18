import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
train = pd.read_csv('../datasets/new/train.csv')
print(train.head())

# Create a directory to save the plots
output_dir = "EDA_plots"
os.makedirs(output_dir, exist_ok=True)

# Define a color palette for better visual appeal
palette = 'Set3'

# List of categorical columns for box plots
categorical_cols = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 
                    'Type_of_vehicle', 'Festival', 'City']

# Loop through each categorical column and create a box plot, save to the EDA_plots folder
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y='Time_taken(min)', data=train, palette=palette)
    plt.title(f'Box plot of Time_taken(min) by {col}', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplot_{col}.png'))  # Save the figure
    plt.close()  # Close the figure to avoid display

# Pair plot for numerical columns, save it to the folder
sns.pairplot(train)
plt.savefig(os.path.join(output_dir, 'pairplot.png'))
plt.close()

# Plot histograms for all numeric features and save each plot
numeric_features = train.select_dtypes(include=['float64', 'int64']).columns
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(train[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histogram_{feature}.png'))  # Save the figure
    plt.close()

# Loop through each categorical column and create a violin plot, save each plot
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=col, y='Time_taken(min)', data=train)
    plt.title(f'Violin plot of Time_taken(min) by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'violinplot_{col}.png'))  # Save the figure
    plt.close()
