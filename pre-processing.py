import pandas as pd

# Load the datasets
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')


# Inspect the train dataset
print("Train Dataset Info:")
print(train.info())
print("\nTrain Dataset Description:")
print(train.describe())
print("\nMissing Values in Train Dataset:")
print(train.isnull().sum())

# Inspect the test dataset
print("\nTest Dataset Info:")
print(test.info())
print("\nTest Dataset Description:")
print(test.describe())
print("\nMissing Values in Test Dataset:")
print(test.isnull().sum())