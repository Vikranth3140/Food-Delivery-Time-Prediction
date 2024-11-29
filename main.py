import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np

def preprocess_and_train(data):
    # Separate features and target
    X = data.drop(columns=['Time_taken(min)'])
    y = data['Time_taken(min)']
    
    # Encode categorical variables
    categorical_features = X.select_dtypes(include='object').columns
    encoders = {col: LabelEncoder() for col in categorical_features}
    for col in categorical_features:
        X[col] = encoders[col].fit_transform(X[col])
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the LightGBM model
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features.tolist())
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    return model, encoders

file_path = 'datasets/kbest features/kbest_features.csv'
data = pd.read_csv(file_path)

# Train the model
model, encoders = preprocess_and_train(data)

def main():
    Road_traffic_density = input("Enter Road Traffic Density (e.g., High, Medium, Low): ")
    Festival = input("Is it during a Festival? (Yes/No): ")
    multiple_deliveries = int(input("Enter number of Multiple Deliveries (e.g., 0, 1): "))
    Delivery_person_Ratings = float(input("Enter Delivery Person's Rating (e.g., 4.5): "))
    Delivery_person_Age = int(input("Enter Delivery Person's Age: "))
    City = input("Enter City Type (e.g., Urban, Metropolitian): ")
    Weatherconditions = input("Enter Weather Conditions (e.g., Sunny, Rainy): ")
    Vehicle_condition = int(input("Enter Vehicle Condition (e.g., 0 for Poor, 1 for Average, 2 for Good): "))
    Type_of_vehicle = input("Enter Type of Vehicle (e.g., motorcycle, scooter): ")
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Road_traffic_density': [Road_traffic_density],
        'Festival': [Festival],
        'multiple_deliveries': [multiple_deliveries],
        'Delivery_person_Ratings': [Delivery_person_Ratings],
        'Delivery_person_Age': [Delivery_person_Age],
        'City': [City],
        'Weatherconditions': [Weatherconditions],
        'Vehicle_condition': [Vehicle_condition],
        'Type_of_vehicle': [Type_of_vehicle]
    })
    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[col])
    
    # Predict using the trained model
    prediction = model.predict(input_data)[0]
    print(f"Predicted Time Taken (minutes): {np.round(prediction, 2)}")

if __name__ == "__main__":
    main()
