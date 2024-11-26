{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Implementing Light GBM model for the dataset with different no. of estimators to find the optimal model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "train = pd.read_csv(\"../../Datasets/new/train.csv\")\n",
    "test = pd.read_csv(\"../../Datasets/new/test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 11/11 [00:00<00:00, 135.81it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Remaining non-numeric columns after encoding: Index([], dtype='object')\n",
      "[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.001221 seconds.\n",
      "You can set `force_row_wise=true` to remove the overhead.\n",
      "And if memory is not enough, you can set `force_col_wise=true`.\n",
      "[LightGBM] [Info] Total Bins 2019\n",
      "[LightGBM] [Info] Number of data points in the train set: 33094, number of used features: 19\n",
      "[LightGBM] [Info] Start training from score 26.520638\n",
      "Number of estimators: 500\n",
      "R-squared: 0.8076\n",
      "Mean Absolute Error: 3.2771\n",
      "Mean Squared Error: 16.8821\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from lightgbm import LGBMRegressor\n",
    "import time\n",
    "\n",
    "#Label Encoding for categorical columns\n",
    "label_encoder = LabelEncoder()\n",
    "\n",
    "categorical_cols = ['Weatherconditions', 'Road_traffic_density', \n",
    "                    'Type_of_order', 'Type_of_vehicle', 'Festival', 'City',\n",
    "                    'ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd',\n",
    "                    'Time_Order_picked']\n",
    "s\n",
    "for col in tqdm(categorical_cols):\n",
    "    train[col] = label_encoder.fit_transform(train[col])\n",
    "\n",
    "#Checking for any non-numeric columns left\n",
    "non_numeric_columns_after = train.select_dtypes(include=['object']).columns\n",
    "print(f\"Remaining non-numeric columns after encoding: {non_numeric_columns_after}\")\n",
    "\n",
    "#x and y labels\n",
    "X_train = train.drop(columns=['Time_taken(min)']) \n",
    "y_train = train['Time_taken(min)'] \n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)\n",
    "\n",
    "#lightGBM model\n",
    "lgbm_model = LGBMRegressor(n_estimators=500, random_state=42)\n",
    "\n",
    "lgbm_model.fit(X_train, y_train)\n",
    "\n",
    "y_pred = lgbm_model.predict(X_test)\n",
    "\n",
    "#Evaluation metrices\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "\n",
    "print(f\"Number of estimators: {500}\")\n",
    "print(f\"R-squared: {r2:.4f}\")\n",
    "print(f\"Mean Absolute Error: {mae:.4f}\")\n",
    "print(f\"Mean Squared Error: {mse:.4f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.11.0 64-bit",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "aee8b7b246df8f9039afb4144a1f6fd8d2ca17a180786b69acc140d282b71a49"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
