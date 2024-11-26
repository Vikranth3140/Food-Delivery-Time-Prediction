import pandas as pd
import os

os.makedirs("../Datasets/new", exist_ok=True)


def clean_dataset(df, is_train=True):
    # Strip white spaces
    df[df.select_dtypes(["object"]).columns] = df.select_dtypes(["object"]).apply(
        lambda x: x.str.strip()
    )

    # Convert columns to appropriate datatypes
    df["Delivery_person_Age"] = pd.to_numeric(
        df["Delivery_person_Age"], errors="coerce"
    ).astype("Int64")
    df["Weatherconditions"] = df["Weatherconditions"].str.replace(
        "conditions ", "", regex=False
    )

    if is_train:
        # Drop "(min)" and convert to integer
        df["Time_taken(min)"] = (
            df["Time_taken(min)"].str.extract("(\d+)").astype("Int64")
        )

    # String columns
    string_columns = [
        "ID",
        "Delivery_person_ID",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
        "Order_Date",
        "Time_Orderd",
        "Time_Order_picked",
    ]
    df[string_columns] = df[string_columns].astype(str)

    # Float columns
    df["Restaurant_latitude"] = pd.to_numeric(
        df["Restaurant_latitude"], errors="coerce"
    )
    df["Restaurant_longitude"] = pd.to_numeric(
        df["Restaurant_longitude"], errors="coerce"
    )
    df["Delivery_location_latitude"] = pd.to_numeric(
        df["Delivery_location_latitude"], errors="coerce"
    )
    df["Delivery_location_longitude"] = pd.to_numeric(
        df["Delivery_location_longitude"], errors="coerce"
    )

    # Numeric columns
    df["Vehicle_condition"] = pd.to_numeric(
        df["Vehicle_condition"], errors="coerce"
    ).astype("Int64")
    df["multiple_deliveries"] = pd.to_numeric(
        df["multiple_deliveries"], errors="coerce"
    ).astype("Int64")

    # Date and time columns
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"], format="%d-%m-%Y", errors="coerce"
    )
    df["Time_Orderd"] = pd.to_datetime(
        df["Time_Orderd"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    df["Time_Order_picked"] = pd.to_datetime(
        df["Time_Order_picked"], format="%H:%M:%S", errors="coerce"
    ).dt.time

    # Drop rows with NaN values
    df = df.replace("NaN", pd.NA)
    df_cleaned = df.dropna()

    return df_cleaned


# Load the train dataset
train = pd.read_csv("../Datasets/kaggle/train.csv", skipinitialspace=True)
train_cleaned = clean_dataset(train, is_train=True)
train_cleaned.to_csv("../Datasets/new/train.csv", index=False)

# Load the test dataset
test = pd.read_csv("../Datasets/kaggle/test.csv", skipinitialspace=True)
test_cleaned = clean_dataset(test, is_train=False)
test_cleaned.to_csv("../Datasets/new/test.csv", index=False)

print("Train dataset cleaned and saved to '../Datasets/new/train.csv'")
print(f"\nNumber of rows in train dataset before cleaning: {train.shape[0]}")
print(f"Number of rows in train dataset after cleaning: {train_cleaned.shape[0]}")

print("Test dataset cleaned and saved to '../Datasets/new/test.csv'")
print(f"\nNumber of rows in test dataset before cleaning: {test.shape[0]}")
print(f"Number of rows in test dataset after cleaning: {test_cleaned.shape[0]}")
