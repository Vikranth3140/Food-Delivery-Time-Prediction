import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
train = pd.read_csv('../datasets/new/train.csv')

# Create a directory to save the plots
output_dir = "EDA_plots"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Step 1: Basic Info and Summary Statistics
print("Basic Info:")
print(train.info())
print("\nSummary Statistics:")
print(train.describe())

# Step 2: Checking Missing Values
print("\nMissing Values:")
print(train.isnull().sum())

# Step 3: Visualizing the Distribution of the Target Variable ('Time_taken(min)')
plt.figure(figsize=(8, 6))
sns.histplot(train['Time_taken(min)'], kde=True, bins=30)
plt.title('Distribution of Time_taken(min)')
plt.xlabel('Time_taken(min)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'Time_taken_distribution.png'))  # Save the figure
plt.close()  # Close the figure

# Step 4: Visualizing Categorical Features
# Plotting some categorical features (e.g., 'Weatherconditions', 'Road_traffic_density')
plt.figure(figsize=(12, 5))
sns.countplot(x='Weatherconditions', data=train)
plt.title('Weatherconditions Count')
plt.savefig(os.path.join(output_dir, 'Weatherconditions_count.png'))  # Save the figure
plt.close()  # Close the figure

plt.figure(figsize=(12, 5))
sns.countplot(x='Road_traffic_density', data=train)
plt.title('Road Traffic Density Count')
plt.savefig(os.path.join(output_dir, 'Road_traffic_density_count.png'))  # Save the figure
plt.close()

# Step 5: Analyzing Numerical Features
# Plot histograms for all numeric features
numeric_features = train.select_dtypes(include=['float64', 'int64']).columns

for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(train[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))  # Save the figure
    plt.close()  # Close the figure

# Step 6: Correlation Analysis (only for numerical columns)
numeric_columns = train.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns
correlation_matrix = train[numeric_columns].corr()  # Calculate correlation matrix only for numeric columns

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numeric Features')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))  # Save the correlation heatmap
plt.close()  # Close the figure

# Step 7: Outlier Detection
# Boxplot to identify outliers in 'Time_taken(min)'
plt.figure(figsize=(8, 6))
sns.boxplot(y='Time_taken(min)', data=train)
plt.title('Boxplot for Time_taken(min)')
plt.savefig(os.path.join(output_dir, 'Time_taken_boxplot.png'))  # Save the boxplot
plt.close()  # Close the figure

# Boxplots for numerical features to detect outliers
for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=feature, data=train)
    plt.title(f'Boxplot for {feature}')
    plt.savefig(os.path.join(output_dir, f'{feature}_boxplot.png'))  # Save the figure
    plt.close()  # Close the figure
