import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the preprocessed dataset
df = pd.read_csv('../datasets/new/train.csv')

# Prepare the features (X) and the target (y)
X = df.drop(columns=['Time_taken(min)'])
y = df['Time_taken(min)']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Apply label encoding on categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Split the dataset into 80% train and 20% validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store results
results = []

# Loop through different numbers of estimators
for n_estimators in tqdm(range(100, 1001, 100), desc="Training Random Forest"):
    
    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    # Train the Random Forest model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Evaluate the model
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    # Store the results
    results.append({
        'n_estimators': n_estimators,
        'r2_score': r2,
        'mean_absolute_error': mae,
        'mean_squared_error': mse
    })

    # Print the evaluation metrics
    print(f"Number of estimators: {n_estimators}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("-" * 50)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot results (R² score vs. number of estimators)
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['r2_score'], marker='o')
plt.title("R² Score vs Number of Estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("R² Score")
plt.grid(True)
plt.show()

# Plot results (MAE vs. number of estimators)
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['mean_absolute_error'], marker='o', color='red')
plt.title("MAE vs Number of Estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("Mean Absolute Error (MAE)")
plt.grid(True)
plt.show()
