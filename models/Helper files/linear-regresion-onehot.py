# Output

# One-Hot Encoding: 100%|████████████████████████████████████████████████████████| 43120/43120 [00:01<00:00, 23765.36it/s]
# Starting to train the Linear Regression model
# Starting to make predictions
# R² Score: 0.59
# Mean Absolute Error (MAE): 4.75
# Total time taken for training and predictions: 7033.54 seconds
#        Actual  Predicted
# 34848      17  15.304211
# 10630      22   9.355344
# 41107      32  35.016605
# 9163       22  21.334744
# 5035       41  36.983238

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import time

# Start the timer
start_time = time.time()

# Load the preprocessed train dataset
train = pd.read_csv("../../Datasets/new/train.csv")

# # Use only the first 100 rows
# train = train.head(100)

# Prepare the features (X) and the target (y)
# Drop the 'Time_taken(min)' column (target) from features
X = train.drop(columns=["Time_taken(min)"])

# Target variable
y = train["Time_taken(min)"]

# One-hot encode categorical variables (dummy encoding) with tqdm for progress tracking
tqdm.pandas(desc="One-Hot Encoding")
X = pd.get_dummies(X, drop_first=True).progress_apply(lambda x: x)

# Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model with all CPU cores
print("Starting to train the Linear Regression model")
model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)

# End the timer for training
end_time = time.time()

# Make predictions on the validation set
print("Starting to make predictions")
y_pred = model.predict(X_val)

# Evaluate the model
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

# Print the evaluation metrics
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Print timing information
print(
    f"Total time taken for training and predictions: {end_time - start_time:.2f} seconds"
)

# Check the few predictions vs actual values
predicted_vs_actual = pd.DataFrame({"Actual": y_val, "Predicted": y_pred})
print(predicted_vs_actual.head())
