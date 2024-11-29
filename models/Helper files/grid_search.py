import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("../content/train.csv")

# Separate features and target variable
X = df.drop(columns=["Time_taken(min)"])
y = df["Time_taken(min)"]

# Encode categorical features
categorical_columns = X.select_dtypes(include=["object"]).columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Normalize numerical features for SVM (important for optimal performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],            # Regularization parameter
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
    'epsilon': [0.1, 0.2, 0.5, 1]     # Epsilon in the loss function
}

# Initialize GridSearchCV with SVR and cross-validation
grid_search = GridSearchCV(SVR(kernel="rbf"), param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearch
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best parameters
best_svm_model = grid_search.best_estimator_

# Make predictions on the validation set
y_pred = best_svm_model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
accuracy = 1 - mse / y_val.var()  # Approximation of accuracy for regression

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Accuracy: {accuracy:.2%}")
