import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
from colorama import init, Fore, Style

def begin_cli():
    init()
    os.system("cls" if os.name == "nt" else "clear")
    title = (
        f"{Fore.CYAN}{Style.BRIGHT}\n"
        " _____               _   ____       _ _                                 \n"
        "|  ___|__   ___   __| | |  _ \\  ___| (_)_   _____ _ __ _   _            \n"
        "| |_ / _ \\ / _ \\ / _` | | | | |/ _ \\ | \\ \\ / / _ \\ '__| | | |           \n"
        "|  _| (_) | (_) | (_| | | |_| |  __/ | |\\ V /  __/ |  | |_| |           \n"
        "|_|  \\___/ \\___/ \\__,_| |____/ \\___|_|_| \\_/ \\___|_|   \\__, |           \n"
        " _____ _                  ____               _ _      _|___/            \n"
        "|_   _(_)_ __ ___   ___  |  _ \\ _ __ ___  __| (_) ___| |_(_) ___  _ __  \n"
        "  | | | | '_ ` _ \\ / _ \\ | |_) | '__/ _ \\/ _` | |/ __| __| |/ _ \\| '_ \\ \n"
        "  | | | | | | | | |  __/ |  __/| | |  __/ (_| | | (__| |_| | (_) | | | |\n"
        " _|_| |_|_| |_| |_|\\___| |_|   |_|  \\___|\\__,_|_|\\___|\\__|_|\\___/|_| |_|\n"
        "/ ___| _   _ ___| |_ ___ _ __ ___                                       \n"
        "\\___ \\| | | / __| __/ _ \\ '_ ` _ \\                                      \n"
        " ___) | |_| \\__ \\ ||  __/ | | | | |                                     \n"
        "|____/ \\__, |___/\\__\\___|_| |_| |_|                                     \n"
        "       |___/                                                            \n"
        f"{Style.RESET_ALL}"
    )
    print(title)

def preprocess_and_train(data):
    X = data.drop(columns=['Time_taken(min)'])
    y = data['Time_taken(min)']
    
    categorical_features = X.select_dtypes(include='object').columns
    encoders = {col: LabelEncoder() for col in categorical_features}
    for col in categorical_features:
        X[col] = encoders[col].fit_transform(X[col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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

# data = pd.read_csv("Datasets/new/train.csv")
data = pd.read_csv("Datasets/kbest features/kbest_features.csv")

model, encoders = preprocess_and_train(data)

def main():
    begin_cli()
    
    def get_input(prompt, cast_type=str, options=None, default=None):
        while True:
            try:
                value = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL} ")
                if not value and default is not None:
                    return default
                if options and value not in options:
                    print(f"{Fore.RED}Invalid choice. Please select from {options}.{Style.RESET_ALL}")
                    continue
                return cast_type(value)
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please enter a valid {cast_type.__name__}.{Style.RESET_ALL}")
    
    Road_traffic_density = get_input(
        "Enter Road Traffic Density (High, Medium, Low):",
        options=['High', 'Medium', 'Low']
    )
    Festival = get_input(
        "Is it during a Festival? (Yes/No):",
        options=['Yes', 'No']
    )
    multiple_deliveries = get_input(
        "Enter number of Multiple Deliveries (e.g., 0, 1):",
        cast_type=int
    )
    Delivery_person_Ratings = get_input(
        "Enter Delivery Person's Rating (e.g., 4.5):",
        cast_type=float
    )
    Delivery_person_Age = get_input(
        "Enter Delivery Person's Age:",
        cast_type=int
    )
    City = get_input(
        "Enter City Type (Urban, Metropolitian):",
        options=['Urban', 'Metropolitian']
    )
    Weatherconditions = get_input(
        "Enter Weather Conditions (Sunny, Cloudy):",
        options=['Sunny', 'Cloudy']
    )
    Vehicle_condition = get_input(
        "Enter Vehicle Condition (0 for Poor, 1 for Average, 2 for Good):",
        cast_type=int,
        options=['0', '1', '2']
    )
    Type_of_vehicle = get_input(
        "Enter Type of Vehicle (motorcycle, scooter):",
        options=['motorcycle', 'scooter']
    )
    
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
    
    prediction = model.predict(input_data)[0]
    print(f"{Fore.GREEN}Predicted Time Taken (minutes): {np.round(prediction, 2)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
