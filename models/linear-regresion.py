import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load train dataset
train = pd.read_csv('../datasets/new/train.csv')

# Prepare the features (X) and the target (y)
X = train.drop(columns=['Time_taken(min)'])

# Target variable
y = train['Time_taken(min)']

# One-hot encode categorical variables (dummy encoding)
tqdm.pandas(desc="One-Hot Encoding")
X = pd.get_dummies(X, drop_first=True).progress_apply(lambda x: x)

# Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
print("Starting to train the Linear Regression model")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
print("Starting to make predictions")
y_pred = model.predict(X_val)

# Evaluate the model
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

# Print the evaluation metrics
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Check the few predictions vs actual values
predicted_vs_actual = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})
print(predicted_vs_actual.head())