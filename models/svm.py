import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# df = pd.read_csv("../Datasets/new/train.csv")
df = pd.read_csv("../Datasets/kbest features/kbest_features.csv")

X = df.drop(columns=["Time_taken(min)"])
y = df["Time_taken(min)"]

categorical_columns = X.select_dtypes(include=["object"]).columns

label_encoder = LabelEncoder()

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm_model = SVR(kernel="rbf")
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_val)

r2 = r2_score(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
accuracy = 1 - mse / y_val.var()

print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Accuracy: {accuracy:.2%}")
