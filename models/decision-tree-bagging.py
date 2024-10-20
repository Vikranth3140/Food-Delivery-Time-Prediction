import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('../datasets/new/train.csv')

# Step 1: Prepare features (X) and target (y)
X = df.drop(columns=['Time_taken(min)'])  # Features (all columns except the target)
y = df['Time_taken(min)']  # Target (Time taken)

# Step 2: Label encode categorical variables
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns  # Select only categorical columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Apply label encoding to each categorical column
    label_encoders[col] = le  # Store the label encoder for potential future use

# Step 3: Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the base model (Decision Tree)
base_model = DecisionTreeRegressor(random_state=42)

# Step 5: Initialize Bagging Regressor
bagging_model = BaggingRegressor(estimator=base_model, n_estimators=50, random_state=42, n_jobs=-1)

# Step 6: Train the Bagging model
bagging_model.fit(X_train, y_train)

# Step 7: Make predictions on the validation set
y_pred = bagging_model.predict(X_val)

# Step 8: Evaluate the model
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Optional: Plot the first few actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_val)), y_val, label="Actual", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6)
plt.title("Bagging: Actual vs Predicted Delivery Time")
plt.xlabel("Samples")
plt.ylabel("Time Taken (min)")
plt.legend()
plt.show()
