# Graduate Employment Status — Classification

Machine Learning course project for predicting graduate employment status using supervised classification models.

## Project Overview

This project predicts **`Employment_Status`** across three classes:

- **Employed**
- **Continuing Education**
- **Unemployed**

The notebook follows a simple, ordered workflow:

**data check → exploratory analysis → feature engineering → train/validation/test split → preprocessing → model comparison → tuning → final evaluation → interpretation**

## Objective

Build and compare machine learning classification models and select the final model using **Validation Macro F1**.

`Macro F1` is used as the main evaluation metric because the three employment classes should receive equal importance.

## Dataset

The notebook expects the dataset file to be available as:

```text
dataset.csv
```

Dataset source: https://www.kaggle.com/datasets/quackquackrp/international-graduates-employment-dataset

The target variable is:

```text
Employment_Status
```

### Features Used

The classification task uses academic, language, visa, university, demographic, and internship information.

The model uses the following features:

- `Country_of_Origin`
- `Education_Level`
- `Field_of_Study`
- `Language_Proficiency`
- `Visa_Type`
- `Gender`
- `University_Ranking`
- `Region_of_Study`
- `Age`
- `Years_Since_Graduation`
- `GPA`
- `Internship_Experience`
- `Education_Language`
- `Education_Internship`
- `University_Internship`

### Excluded Features

- `Salary` — excluded from classification and used later for salary regression.
- `Job_Sector` — excluded because it describes the job sector and contains many missing values for graduates without a job.

## Feature Engineering

Three interaction features are created from patterns observed during exploratory analysis:

- `Education_Language`
- `Education_Internship`
- `University_Internship`

## Data Preparation

The data is split using stratified sampling into:

- **80% Training**
- **10% Validation**
- **10% Test**

Categorical variables are processed using **One-Hot Encoding**, while numerical variables are scaled using **StandardScaler**.

The preprocessing transformer is fitted only on the training data.

## Models

The project compares the following classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- HistGradientBoosting
- Linear SVM

A **Dummy Classifier** is also used as a simple baseline.

Some models are tuned using the validation set.

## Evaluation

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- **Macro F1**
- Confusion Matrix
- ROC-AUC (when probability estimates are available)

The final model is selected using **Validation Macro F1**. The Test set is then used only for final performance reporting.

## Main Findings

The current results show useful predictive signal for employment status, while **Continuing Education** remains the hardest class to distinguish.

The analysis identifies strong signals related to:

- Education level
- Language proficiency
- GPA
- Years since graduation
- Selected education, internship, and university interaction features

## Interpretation

Feature interpretation is performed separately from final model selection.

A Decision Tree is used to inspect:

- Built-in feature importance
- Permutation importance

Permutation importance is calculated using **Validation Macro F1** to estimate the contribution of the original features.

## Final Results

In the current notebook run, the validation results are used to select the final predictive model, and the selected model is evaluated once on the test set.

The notebook reports the complete model-by-model results, classification report, confusion matrix, and ROC-AUC analysis.

## Project Files

```text
.
├── README.md
├── graduate_employment_classification.ipynb
└── dataset.csv
```

> `dataset.csv` is required to run the notebook because the notebook loads it directly with `pd.read_csv("dataset.csv")`.

## How to Run

1. Place `dataset.csv` in the same directory as the notebook.
2. Open `graduate_employment_classification.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the cells from top to bottom.

## Course Project

This repository contains a **Machine Learning course project** demonstrating data exploration, feature engineering, preprocessing, model comparison, hyperparameter tuning, evaluation, and model interpretation.
