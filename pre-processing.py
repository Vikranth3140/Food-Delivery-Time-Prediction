import pandas as pd

# Load the dataset
train = pd.read_csv('datasets/train.csv', skipinitialspace=True)

train = train.applymap(lambda x: x.strip() if isinstance(x, str) else x)

train['Delivery_person_Age'] = pd.to_numeric(train['Delivery_person_Age'], errors='coerce').astype('Int64')
train['Weatherconditions'] = train['Weatherconditions'].str.replace("conditions ", "", regex=False)
train['Time_taken(min)'] = train['Time_taken(min)'].str.extract('(\d+)').astype('Int64')

string_columns = [
    'ID', 'Delivery_person_ID', 'Road_traffic_density', 'Type_of_order', 
    'Type_of_vehicle', 'Festival', 'City', 'Order_Date', 'Time_Orderd', 
    'Time_Order_picked'
]
train[string_columns] = train[string_columns].astype(str)

train['Restaurant_latitude'] = pd.to_numeric(train['Restaurant_latitude'], errors='coerce')
train['Restaurant_longitude'] = pd.to_numeric(train['Restaurant_longitude'], errors='coerce')
train['Delivery_location_latitude'] = pd.to_numeric(train['Delivery_location_latitude'], errors='coerce')
train['Delivery_location_longitude'] = pd.to_numeric(train['Delivery_location_longitude'], errors='coerce')

train['Vehicle_condition'] = pd.to_numeric(train['Vehicle_condition'], errors='coerce').astype('Int64')
train['multiple_deliveries'] = pd.to_numeric(train['multiple_deliveries'], errors='coerce').astype('Int64')

train['Order_Date'] = pd.to_datetime(train['Order_Date'], format='%d-%m-%Y', errors='coerce')
train['Time_Orderd'] = pd.to_datetime(train['Time_Orderd'], format='%H:%M:%S', errors='coerce').dt.time
train['Time_Order_picked'] = pd.to_datetime(train['Time_Order_picked'], format='%H:%M:%S', errors='coerce').dt.time

# Drop rowa with NaN values
train_cleaned = train.dropna()

print(train_cleaned.head())
print("\nData types after dropping NaNs:")
print(train_cleaned.dtypes)

print(f"\nNumber of rows before dropping NaNs: {train.shape[0]}")
print(f"Number of rows after dropping NaNs: {train_cleaned.shape[0]}")