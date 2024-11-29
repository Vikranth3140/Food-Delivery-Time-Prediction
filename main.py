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


def get_user_input():
    print(
        f"\n{Fore.YELLOW}Please enter the details to predict the time taken for delivery.{Style.RESET_ALL}\n"
    )
    return {
        "Road_traffic_density": input(
            f"{Fore.CYAN}Enter Road Traffic Density (High, Medium, Low): {Style.RESET_ALL}"
        ),
        "Festival": input(
            f"{Fore.CYAN}Is it during a Festival? (Yes/No): {Style.RESET_ALL}"
        ),
        "multiple_deliveries": int(
            input(
                f"{Fore.CYAN}Enter number of Multiple Deliveries (e.g., 0, 1): {Style.RESET_ALL}"
            )
        ),
        "Delivery_person_Ratings": float(
            input(
                f"{Fore.CYAN}Enter Delivery Person's Rating (e.g., 4.5): {Style.RESET_ALL}"
            )
        ),
        "Delivery_person_Age": int(
            input(f"{Fore.CYAN}Enter Delivery Person's Age: {Style.RESET_ALL}")
        ),
        "City": input(
            f"{Fore.CYAN}Enter City Type (Urban, Metropolitian): {Style.RESET_ALL}"
        ),
        "Weatherconditions": input(
            f"{Fore.CYAN}Enter Weather Conditions (Sunny, Cloudy): {Style.RESET_ALL}"
        ),
        "Vehicle_condition": int(
            input(
                f"{Fore.CYAN}Enter Vehicle Condition (0 for Poor, 1 for Average, 2 for Good): {Style.RESET_ALL}"
            )
        ),
        "Type_of_vehicle": input(
            f"{Fore.CYAN}Enter Type of Vehicle (motorcycle, scooter): {Style.RESET_ALL}"
        ),
    }


def predict_time(input_data, model, encoders):
    input_df = pd.DataFrame([input_data])
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    prediction = model.predict(input_df)[0]
    return np.round(prediction, 2)


def predict_range(time):
    if time <= 15:
        return f"Very Quick (<= 15 minutes)"
    elif time <= 30:
        return f"Quick (15 - 30 minutes)"
    elif time <= 45:
        return f"Moderate (30 - 45 minutes)"
    elif time <= 60:
        return f"Slow (45 - 60 minutes)"
    else:
        return f"Very Slow (>= 60 minutes)"


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


if __name__ == "__main__":
    begin_cli()
    print(f"\n{Fore.YELLOW}Welcome to the Delivery Time Predictor CLI!{Style.RESET_ALL}\n")

    # data = pd.read_csv("Datasets/new/train.csv")
    data = pd.read_csv("Datasets/kbest features/kbest_features.csv")

    model, encoders = preprocess_and_train(data)

    while True:
        user_input = get_user_input()
        predicted_time = predict_time(user_input, model, encoders)
        predicted_range = predict_range(predicted_time)

        print(f"\n{Fore.GREEN}Predicted Time Taken{Style.RESET_ALL}: {predicted_time} minutes")
        print(f"{Fore.GREEN}Delivery Speed Range{Style.RESET_ALL}: {predicted_range}\n")

        another = input(
            f"{Fore.YELLOW}Would you like to predict another delivery? (yes/no): {Style.RESET_ALL}"
        ).lower()
        if another != "yes":
            break

    print(
        f"\n{Fore.YELLOW}Thank you for using the Delivery Time Prediction System!{Style.RESET_ALL}"
    )
