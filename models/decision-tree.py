import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('../datasets/new/train.csv')

# Step 1: Prepare features (X) and target (y)
X = df.drop(columns=['Time_taken(min)'])  # Features (all columns except the target)
y = df['Time_taken(min)']  # Target (Time taken)

# Step 2: One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Step 3: Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions on the validation set
y_pred = model.predict(X_val)

# Step 7: Evaluate the model
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Optional: Plot the first few actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_val)), y_val, label="Actual", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6)
plt.title("Decision Tree: Actual vs Predicted Delivery Time")
plt.xlabel("Samples")
plt.ylabel("Time Taken (min)")
plt.legend()
plt.show()
