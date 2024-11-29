# Food Delivery Time Prediction

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![sk-learn](https://img.shields.io/badge/scikit-learn-grey.svg?logo=scikit-learn)](https://scikit-learn.org/stable/whats_new.html)

<img src="fig/image.png" alt="Food Delivery Illustration" width="500" height="400">

## Introduction

Running a food delivery service comes with the challenge of keeping customers happy by delivering their meals on time and in condition despite hurdles like traffic or bad weather which can throw off the schedule unpredictably.

In order to address this issue effectively we are working on a Food Delivery Time Prediction System that utilizes machine learning methods. Our goal is to predict delivery times with precision by examining delivery data, current traffic situations and real time weather trends.

We have developed a Command Line Interface (CLI) to allow users to input food delivery parameters and get delivery time predictions. This tool provides an estimate of the inputted food delivery time within specific ranges:

- Very Quick: <= 15 minutes
- Quick: 15 - 30 minutes
- Moderate: 30 - 45 minutes
- Slow: 45 - 60 minutes
- Very Slow: >= 60 minutes


## Directory Structure

```
Food-Delivery-Time-Prediction/
│
├── Data Pre Processing/
│   └── pre-processing.py
│
├── datasets/
│   ├── kaggle/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── README.md
│   │
│   ├── kbest features/
│   │   └── kbest_features.csv
│   │
│   └── new/
│       ├── train.csv
│       └── test.csv
│
├── EDA/
│   ├── EDA_plots/
│   │   ├── boxplot_City.png
│   │   ├── boxplot_Festival.png
│   │   ├── boxplot_Road_traffic_density.png
│   │   ├── boxplot_Type_of_order.png
│   │   ├── boxplot_Type_of_vehicle.png
│   │   ├── boxplot_Weatherconditions.png
│   │   ├── histogram_Delivery_location_latitude.png
│   │   ├── histogram_Delivery_location_longitude.png
│   │   ├── histogram_Delivery_person_Age.png
│   │   ├── histogram_Delivery_person_Rating.png
│   │   ├── histogram_Distance.png
│   │   ├── histogram_Multiple_deliveries.png
│   │   ├── histogram_Restaurant_latitude.png
│   │   ├── histogram_Restaurant_longitude.png
│   │   ├── histogram_Time_taken(min).png
│   │   ├── histogram_Vehicle_condition.png
│   │   ├── pairplot.png
│   │   ├── umap_projection.png
│   │   ├── violinplot_City.png
│   │   ├── violinplot_Festival.png
│   │   ├── violinplot_Road_traffic_density.png
│   │   ├── violinplot_Type_of_order.png
│   │   ├── violinplot_Type_of_vehicle.png
│   │   └── violinplot_Weatherconditions.png
│   │
│   ├── analysis.py
│   └── umap_and_heatmap.ipynb
│
├── Feature Selection/
│   ├── Helper Files/
│   │   └── random_forest_importance.py
│   │
│   ├── aggregated_feature_scores.csv
│   ├── feature_scores.txt
│   ├── kbest.py
│   └── save_kbest_features.py
│
├── fig/
│   └── image.png
│
├── models/
│   ├── Helper files/
│   │   ├── elasticnetRegularization.ipynb
│   │   ├── k_cross_validation.py
│   │   ├── lightGBM.ipynb
│   │   ├── linear-regression-onehot.py
│   │   ├── linear-regression.ipynb
│   │   ├── random-forest.ipynb
│   │   ├── SVM.ipynb
│   │   └── xgboost.py
│   │
│   ├── Plots/
│   │   ├── decision-tree-bagging.png
│   │   ├── decision-tree.png
│   │   ├── k_cross_validation.png
│   │   ├── linear-regression.png
│   │   ├── random-forest-2.png
│   │   └── random-forest.png
│   │
│   ├── accuracies.txt
│   ├── decision-tree-bagging.py
│   ├── decision-tree.py
│   ├── elasticnetRegularization.py
│   ├── k_cross_validation.py
│   ├── lightGBM.py
│   ├── linear-regression.py
│   ├── random-forest.py
│   ├── svm.py
│   └── tracking_XGBoost.py
│
├── Reports/
│   ├── Proposal/
│   │   ├── proposal.pdf
│   │   ├── proposal.tex
│   ├── MidSem Report/
│   │   ├── ML Project MidSem Report.pdf
│   │   ├── ML Project MidSem Project Slides.pdf
│   │   └── ML Project MidSem Report.tex
│   │
│   └── Final Report/
│       ├── final_report.tex
│       ├── final_report.pdf
│       ├── cross_validation.png
│       ├── decision-tree.png
│       ├── decision-tree-bagging.png
│       ├── random-forest.png
│       ├── random-forest-2.png
│       ├── linear-regression.png
│       └── xgboost.png
│
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── main.py
├── predictor-accuracy.py
├── README.md
└── requirements.txt
```

## Getting Started

All our code was tested on Python 3.6.8 with scikit-learn 1.3.2. Ideally, our scripts require access to a single GPU (uses `.cuda()` for inference). Inference can also be done on CPUs with minimal changes to the scripts.

### Setting up the Environment

We recommend setting up a Python virtual environment and installing all the requirements. Please follow these steps to set up the project folder correctly:

```bash
git clone https://github.com/Vikranth3140/Food-Delivery-Time-Prediction.git
cd Food-Delivery-Time-Prediction

python3 -m venv ./env
source env/bin/activate

pip install -r requirements.txt
```

### Setting up Datasets

The datasets have been taken from [Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset/data) dataset.

## Running the Models

You can run the final model using:

```bash
python main.py
```

## Data Preprocessing

The Data has been pre processed:

- Removed 'conditions' prefix from 'Weather Conditions'.
- Standardized columns into appropriate formats: 'strings', 'integers', and 'floats'.
- Converted Order Date to 'datetime' format.
- Extracted time from 'Time Ordered' and 'Time Order Picked'.
- Dropped rows with null values for consistency.

### Encoding Categorical Variables

Categorical variables are encoded using Label Encoding.

### Feature Selection

Some features are dropped in the dataset due to lower scores in [kbest.py](Feature Selection\kbest.py):

- ID
- Delivery_person_ID
- Order_Date
- Time_Orderd
- Time_Order_picked

### Feature Selection

We use SelectKBest for helping us know which features contribute the most towards our target variable as shown in [kbest.py](Feature Selection\kbest.py).

## Model Improvement

We employed strategies such as hyperparameter tuning using `GridSearchCV` for model improvement.

### Hyperparameter Tuning

Hyperparameter tuning is performed using `GridSearchCV` to optimize model parameters.

## Command Line Interface (CLI)

We have developed a `Command Line Interface (CLI)` to allow users to input delivery time parameters and get delivery time predictions. This tool provides an estimate of the inputted food delivery time within specific ranges:

- Very Quick: <= 15 minutes
- Quick: 15 - 30 minutes
- Moderate: 30 - 45 minutes
- Slow: 45 - 60 minutes
- Very Slow: >= 60 minutes

### Using the CLI

1. Navigate to the project directory.
2. Run the CLI:
   ```bash
   python main.py
   ```
3. Follow the prompts to input the movie features and choose the prediction model.

## Model Evaluation Results

We evaluated our models using two key metrics: R² Score (Coefficient of Determination) and MSE (Mean Squared Error). Here are the results for each model:

| Model             | MSE | R² Score |
| ----------------- | ----------- | ------------ |
        Linear Regression              & 42.80     & 0.51
        Decision Tree                  & 41.14     & 0.53
        Decision Tree with Bagging     & 21.67     & 0.75
        Random Forest                  & 21.21     & 0.75
        Elastic Net Regularization     & 47.35     & 0.46
        LightGBM                      & 16.88     & 0.80
        XGBoost                       & 18.41     & 0.79

## Conclusion

The developed LightGBM model demonstrates promising accuracy and generalization capabilities, facilitating informed decision-making in the food delivery space to predict delivery time.

<!-- ## Citation

If you found this work useful, please consider citing it as:

```
@misc{udandarao2024movie,
      title={Movie Revenue Prediction using Machine Learning Models},
      author={Vikranth Udandarao and Pratyush Gupta},
      year={2024},
      eprint={2405.11651},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` -->

## Contact

Please feel free to open an issue or email us at [vikranth22570@iiitd.ac.in](mailto:vikranth22570@iiitd.ac.in), [ayaan22302@iiitd.ac.in](mailto:ayaan22302@iiitd.ac.in), [swara22524@iiitd.ac.in](mailto:swara22524@iiitd.ac.in) or [ananya22068@iiitd.ac.in](mailto:ananya22068@iiitd.ac.in).

## License

This project is licensed under the [MIT License](LICENSE).
