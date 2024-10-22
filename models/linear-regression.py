import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

# Initialize the Linear Regression model
model = LinearRegression()

# Train the Linear Regression model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

# Print the evaluation metrics
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Plot the first few actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_val)), y_val, label="Actual", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6)
plt.title("Linear Regression: Actual vs Predicted Delivery Time")
plt.xlabel("Samples")
plt.ylabel("Time Taken (min)")
plt.legend()
plt.show()
