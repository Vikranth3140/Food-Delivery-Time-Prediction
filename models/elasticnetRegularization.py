import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# df = pd.read_csv("../Datasets/new/train.csv")
df = pd.read_csv("../Datasets/kbest features/kbest_features.csv")

label_encoder = LabelEncoder()

categorical_cols = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_vehicle",
    "Festival",
    "City",
]

for col in tqdm(categorical_cols):
    df[col] = label_encoder.fit_transform(df[col])

non_numeric_columns_after = df.select_dtypes(include=["object"]).columns
print(f"Remaining non-numeric columns after encoding: {non_numeric_columns_after}")

X = df.drop(columns=["Time_taken(min)"])
y = df["Time_taken(min)"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

l1Ratios = [0, 0.1, 0.5, 0.7, 1]
alphaValues = [0.1, 0.3, 0.5, 0.8, 1]

metrices = {
    alpha: {
        "mseTrain": [],
        "mseTest": [],
        "rmseTrain": [],
        "rmseTest": [],
        "r2Train": [],
        "r2Test": [],
        "adjustedR2Train": [],
        "adjustedR2Test": [],
        "maeTrain": [],
        "maeTest": [],
    }
    for alpha in alphaValues
}

for l1_ratio in l1Ratios:
    for alpha in alphaValues:
        print(f"ELASTICNET WITH ALPHA = {alpha} AND L1RATIO = {l1_ratio}")

        elasticNetModel = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=30)
        elasticNetModel.fit(X_train, y_train)

        y_trainPred = elasticNetModel.predict(X_train)
        y_testPred = elasticNetModel.predict(X_val)

        mseTrain_elasticNet = mean_squared_error(y_train, y_trainPred)
        mseTest_elasticNet = mean_squared_error(y_val, y_testPred)

        rmseTrain_elasticNet = np.sqrt(mseTrain_elasticNet)
        rmseTest_elasticNet = np.sqrt(mseTest_elasticNet)

        r2Train_elasticNet = r2_score(y_train, y_trainPred)
        r2Test_elasticNet = r2_score(y_val, y_testPred)

        n = X_train.shape[0]
        f = X_train.shape[1]

        adjustedR2Train_elasticNet = 1 - (1 - r2Train_elasticNet) * (n - 1) / (
            n - f - 1
        )
        adjustedR2Test_elasticNet = 1 - (1 - r2Test_elasticNet) * (n - 1) / (n - f - 1)

        maeTrain_elasticNet = mean_absolute_error(y_train, y_trainPred)
        maeTest_elasticNet = mean_absolute_error(y_val, y_testPred)

        metrices[alpha]["mseTrain"].append(mseTrain_elasticNet)
        metrices[alpha]["mseTest"].append(mseTest_elasticNet)
        metrices[alpha]["rmseTrain"].append(rmseTrain_elasticNet)
        metrices[alpha]["rmseTest"].append(rmseTest_elasticNet)
        metrices[alpha]["r2Train"].append(r2Train_elasticNet)
        metrices[alpha]["r2Test"].append(r2Test_elasticNet)
        metrices[alpha]["adjustedR2Train"].append(adjustedR2Train_elasticNet)
        metrices[alpha]["adjustedR2Test"].append(adjustedR2Test_elasticNet)
        metrices[alpha]["maeTrain"].append(maeTrain_elasticNet)
        metrices[alpha]["maeTest"].append(maeTest_elasticNet)

        print(f"MSE Train: {mseTrain_elasticNet:.4f}")
        print(f"MSE Test: {mseTest_elasticNet:.4f}")
        print(f"RMSE Train: {rmseTrain_elasticNet:.4f}")
        print(f"RMSE Test: {rmseTest_elasticNet:.4f}")
        print(f"R2 Train: {r2Train_elasticNet:.4f}")
        print(f"R2 Test: {r2Test_elasticNet:.4f}")
        print(f"Adjusted R2 Train: {adjustedR2Train_elasticNet:.4f}")
        print(f"Adjusted R2 Test: {adjustedR2Test_elasticNet:.4f}")
        print(f"MAE Train: {maeTrain_elasticNet:.4f}")
        print(f"MAE Test: {maeTest_elasticNet:.4f}")
        print("\n")


def plotFigure(metric, dataset):
    for alpha in alphaValues:
        plt.plot(
            l1Ratios,
            metrices[alpha][f"{metric}{dataset}"],
            marker="o",
            label=f"Alpha = {alpha}",
        )
        plt.xlabel("L1 Ratio")
        plt.ylabel(metric)
        plt.title(f"{metric} vs L1_Ratio")
        plt.legend()


subplots = [1, 2, 3, 4, 5]
values = ["mse", "rmse", "r2", "adjustedR2", "mae"]

plt.figure(figsize=(15, 10))
plt.suptitle("TRAIN DATA")

for i in range(5):
    plt.subplot(2, 3, subplots[i])
    plotFigure(values[i], "Train")

plt.figure(figsize=(15, 10))
plt.suptitle("TEST DATA")

for i in range(5):
    plt.subplot(2, 3, subplots[i])
    plotFigure(values[i], "Test")
