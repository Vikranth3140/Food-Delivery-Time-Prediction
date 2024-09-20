import pandas as pd

# Load the dataset
train = pd.read_csv('datasets/train.csv', skipinitialspace=True)

# Step 1: Convert `Delivery_person_Age` to int64
train['Delivery_person_Age'] = pd.to_numeric(train['Delivery_person_Age'], errors='coerce').astype('Int64')

# Step 2: Clean `Weatherconditions` to drop "conditions"
train['Weatherconditions'] = train['Weatherconditions'].str.replace("conditions ", "", regex=False)

# Step 3: Clean `Time_taken(min)` to drop "(min)" and convert to integer
train['Time_taken(min)'] = train['Time_taken(min)'].str.extract('(\d+)').astype('Int64')

# Step 4: Ensure other columns are strings
string_columns = [
    'ID', 'Delivery_person_ID', 'Road_traffic_density', 'Type_of_order', 
    'Type_of_vehicle', 'Festival', 'City', 'Order_Date', 'Time_Orderd', 
    'Time_Order_picked'
]
train[string_columns] = train[string_columns].astype(str)

# Step 5: Convert Latitude and Longitude to float
train['Restaurant_latitude'] = pd.to_numeric(train['Restaurant_latitude'], errors='coerce')
train['Restaurant_longitude'] = pd.to_numeric(train['Restaurant_longitude'], errors='coerce')
train['Delivery_location_latitude'] = pd.to_numeric(train['Delivery_location_latitude'], errors='coerce')
train['Delivery_location_longitude'] = pd.to_numeric(train['Delivery_location_longitude'], errors='coerce')

# Step 6: Convert `Vehicle_condition` and `multiple_deliveries` to integers
train['Vehicle_condition'] = pd.to_numeric(train['Vehicle_condition'], errors='coerce').astype('Int64')
train['multiple_deliveries'] = pd.to_numeric(train['multiple_deliveries'], errors='coerce').astype('Int64')

# Step 7: Convert `Order_Date` to datetime and `Time_Orderd`, `Time_Order_picked` to time
train['Order_Date'] = pd.to_datetime(train['Order_Date'], format='%d-%m-%Y', errors='coerce')
train['Time_Orderd'] = pd.to_datetime(train['Time_Orderd'], format='%H:%M:%S', errors='coerce').dt.time
train['Time_Order_picked'] = pd.to_datetime(train['Time_Order_picked'], format='%H:%M:%S', errors='coerce').dt.time

# Final check of the processed dataset
print(train.head())
print("\nData types after conversion:")
print(train.dtypes)