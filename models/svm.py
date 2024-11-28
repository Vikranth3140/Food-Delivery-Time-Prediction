import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
# df = pd.read_csv("../Datasets/new/train.csv")
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

# Train an SVM model
svm_model = SVR(kernel="rbf")  # Using RBF kernel for regression
svm_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = svm_model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
accuracy = 1 - mse / y_val.var()  # Approximation of accuracy for regression

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Accuracy: {accuracy:.2%}")
