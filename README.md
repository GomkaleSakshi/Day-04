# Facility Hygiene Risk Prediction

## Objective

This beginner-friendly machine learning project predicts whether a facility is at Low, Medium, or High hygiene risk from routine cleanliness, odor, waste, complaint, footfall, and cleaning-time measurements.

## Problem Statement

Facility teams need a consistent way to identify sites that may need attention. The model uses historical facility measurements to classify hygiene risk. This is a classification problem because `hygiene_risk` contains three categories: Low, Medium, and High.

## Dataset Description

`dataset/facility_hygiene_data.csv` contains at least 100 synthetic facility records. It includes realistic relationships between poor cleanliness, odor, waste, complaints, footfall, time since cleaning, and increased risk. A small number of feature values are missing and several records are duplicated so that the cleaning steps can be demonstrated.

## Features and Target

The six numeric input features are `cleanliness_score`, `odor_score`, `waste_level`, `complaints`, `footfall`, and `hours_since_cleaning`. The target is `hygiene_risk`, with values Low, Medium, or High. The three score fields use a 0-10 scale. Preprocessing adds `cleaning_overdue`, which is 1 when more than eight hours have passed since cleaning.

## ML Concepts Used

The project demonstrates tabular data loading, data quality checks, median imputation, duplicate removal, exploratory data analysis, feature engineering, categorical target encoding, stratified train/test splitting, supervised classification, model comparison, prediction, and evaluation.

## Data Preprocessing

`preprocessing/preprocessing.py` displays dataset shape, information, missing values, data types, descriptive statistics, and risk group summaries. It removes duplicate rows, converts input fields to numeric, treats out-of-range scores and negative counts/times as invalid, fills missing feature values with their medians, adds the overdue-cleaning feature, and encodes Low=0, Medium=1, High=2. The processed data is written to `dataset/processed_hygiene_data.csv`.

## EDA

The evaluation script generates a risk distribution chart and boxplots for cleanliness, odor, waste, and hours since cleaning by risk category. These plots use matplotlib and are saved in `evaluation/`.

## Feature Engineering

`cleaning_overdue` is a simple binary feature derived from `hours_since_cleaning`. It helps the model distinguish facilities that have not been cleaned recently.

## Algorithms Used

- Logistic Regression
- Decision Tree Classifier

Both models are trained on the same processed features and evaluated on the same stratified test set.

## Train/Test Split

The scripts use `train_test_split` with `test_size=0.2` and `random_state=42`; stratification keeps the three risk classes represented in both partitions.

## Evaluation Metrics

Accuracy, weighted precision, weighted recall, weighted F1 score, and confusion matrices are calculated for each model. The confusion matrices are saved as PNG images in `evaluation/`.

## Model Comparison and Selected Model

The detailed metric comparison is generated in `evaluation/model_comparison.txt`. The model with the higher weighted F1 score is selected automatically. If weighted F1 is tied, Logistic Regression is selected as the simpler model to explain. `evaluation/evaluation_results.txt` includes the evaluation details and both confusion matrices. The prediction script reads the selected model name from the comparison file, so it uses the measured best model rather than assuming one in advance.

## Key Findings

The cleaned dataset contains 47 Low, 47 Medium, and 46 High risk facilities. High-risk facilities have lower average cleanliness (2.77) and higher average odor (8.65) than Low-risk facilities (8.70 cleanliness and 1.72 odor). Both models scored 1.0000 for accuracy and weighted F1 on this structured synthetic holdout, so they tie; Logistic Regression is selected because it is simpler to explain. This perfect synthetic-data score is not evidence of real-world performance.

## Problems Encountered

This is a synthetic teaching dataset, not operational facility data. Missing measurements and duplicate records are included intentionally. Median imputation keeps the example simple, while the model scores should not be treated as proof of real-world performance.

## Possible Improvements

With permission to use real, representative data, useful next steps include validating score definitions with facility staff, collecting more examples, testing performance on a later time period, tuning model settings, reviewing class-specific errors, and monitoring predictions for drift.

## Project Structure

```text
Day-04/
|-- dataset/
|   |-- facility_hygiene_data.csv
|   `-- processed_hygiene_data.csv
|-- preprocessing/
|   `-- preprocessing.py
|-- models/
|   |-- train_models.py
|   |-- logistic_regression.joblib
|   `-- decision_tree.joblib
|-- evaluation/
|   |-- evaluate_models.py
|   |-- evaluation_results.txt
|   |-- model_comparison.txt
|   `-- generated EDA and confusion matrix charts
|-- predictions/
|   |-- predict_hygiene.py
|   `-- hygiene_predictions.csv
`-- README.md
```

## Technologies Used

Python, pandas, NumPy, matplotlib, scikit-learn, and joblib.

## Installation

From the Day-04 root folder, install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## How to Run

Run these commands in order from the Day-04 root folder:

```bash
python preprocessing\preprocessing.py
python models\train_models.py
python evaluation\evaluate_models.py
python predictions\predict_hygiene.py
```

The scripts resolve project files relative to their own location, so these commands work from the project root on Windows. On macOS or Linux, use forward slashes in the command paths.

## Expected Output

Preprocessing creates the cleaned CSV. Training saves both models and a metric comparison that names the best model. Evaluation prints metrics, saves evaluation text, and generates EDA and confusion matrix charts. Prediction loads the best model, prints example facility risk classifications, and saves them to `predictions/hygiene_predictions.csv`.
