import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb


def preprocess_and_train(data):
    X = data.drop(columns=["Time_taken(min)"])
    y = data["Time_taken(min)"]

    categorical_features = X.select_dtypes(include="object").columns
    encoders = {col: LabelEncoder() for col in categorical_features}
    for col in categorical_features:
        X[col] = encoders[col].fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features.tolist())
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )

    return model, encoders


# data = pd.read_csv("Datasets/new/train.csv")
data = pd.read_csv("Datasets/kbest features/kbest_features.csv")

model, encoders = preprocess_and_train(data)


def evaluate_accuracy(data, model, encoders, num_samples=200, threshold=7):
    sample_data = data.sample(n=num_samples, random_state=42)
    true_values = sample_data["Time_taken(min)"]
    input_data = sample_data.drop(columns=["Time_taken(min)"])

    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[col])

    predictions = model.predict(input_data)

    correct_predictions = np.abs(predictions - true_values) <= threshold
    accuracy = correct_predictions.mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    accuracy = evaluate_accuracy(data, model, encoders)
