# Bank Marketing Classification

A machine-learning classification project developed for the Markovate assessment **“Building an ML-Based Classification Solution.”** The project uses the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Objective and Target Variable](#objective-and-target-variable)
- [Approach](#approach)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preparation and Key Decisions](#data-preparation-and-key-decisions)
- [Models](#models)
- [Evaluation Results](#evaluation-results)
- [Feature Importance](#feature-importance)
- [Limitations](#limitations)
- [Proposed Improvements](#proposed-improvements)
- [Real-World Application](#real-world-application)
- [AI Tool Usage](#ai-tool-usage)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)

## Problem Statement

Businesses often need to classify incoming records to reduce manual effort and support better decisions. In this project, the classification task is to predict whether a bank customer will subscribe to a term deposit. Such predictions could help a bank prioritize customers for marketing outreach.

The goal is to build and evaluate a basic, reproducible machine-learning classification workflow using Python, exploratory data analysis, preprocessing, and supervised learning.

## Dataset Overview

- **Dataset:** Bank Marketing dataset (`bank.csv`)
- **Source:** Publicly available dataset obtained from Kaggle. Add the exact Kaggle dataset page URL here before submission so the dataset version can be identified.
- **Rows:** 11,162
- **Columns:** 17 in the original dataset
- **Target column:** `deposit`
- **Task:** Binary classification

The target contains two classes:

| Class | Count | Approximate proportion |
|---|---:|---:|
| `no` | 5,873 | 52.62% |
| `yes` | 5,289 | 47.38% |

The target classes are reasonably balanced, so accuracy is useful as one metric, but it should not be used alone.

### Key features

The original dataset includes customer and campaign information such as:

- **Customer characteristics:** `age`, `job`, `marital`, `education`
- **Financial information:** `balance`, `default`, `housing`, `loan`
- **Contact and campaign information:** `contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome`
- **Additional field:** `duration`

The `deposit` column is the target. The `duration` feature is excluded from the model for the reason explained below.

## Objective and Target Variable

The model predicts whether a customer will subscribe to a term deposit:

- `no` → `0`
- `yes` → `1`

The intended prediction point is **before contacting the customer**. Therefore, features should only represent information that could be available at that point.

## Approach

The project follows these steps:

1. Load the dataset and inspect its shape, data types, and sample records.
2. Check for missing values, duplicate rows, unknown category values, and unusual numerical values.
3. Perform exploratory data analysis (EDA) to examine the target distribution and relationships between customer features and subscription outcomes.
4. Separate the target from the input features and exclude `duration`.
5. Split the data into training and test sets, using stratification to preserve the target-class proportions.
6. Preprocess numerical and categorical columns within scikit-learn pipelines.
7. Train Logistic Regression and Random Forest classifiers.
8. Evaluate both models using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
9. Inspect Random Forest feature importance and select a preferred model based on the observed test-set results.
10. Save and reload the trained Random Forest pipeline and verify that its predictions match those of the original pipeline.

## Exploratory Data Analysis

The EDA included the following checks and visualizations:

- Target-class distribution
- Subscription rates across age groups
- Subscription rates by job category
- Subscription rates across balance groups
- Subscription rates by month
- Subscription rates by previous campaign outcome
- Numerical feature distributions and potential outliers

### Data quality observations

- The dataset contained **no missing values** in the original columns.
- No duplicate rows were found in the initial duplicate check.
- Several categorical columns contain the value `unknown`. This was retained as a category rather than automatically treated as missing. In particular, `contact` and `poutcome` have substantial numbers of `unknown` entries.
- Several numerical features are right-skewed and contain potential outliers. These were not automatically removed because extreme values can be legitimate and removing them without context could discard useful information.
- `pdays = -1` is used to represent customers who were not previously contacted; it was retained rather than treated as an ordinary missing value.

EDA relationships are descriptive and should not be interpreted as evidence that a feature causes subscription.

## Data Preparation and Key Decisions

### Train-test split

The dataset was split into training and test subsets using an 80:20 ratio and `stratify=y`. A fixed `random_state=42` was used to make the split reproducible.

- **Training rows:** 8,929
- **Test rows:** 2,233

### Feature selection and leakage prevention

The `deposit` column was separated as the target. The `duration` column was excluded because the project aims to predict subscription before contacting the customer. Call duration would not be known at that prediction point and could introduce target leakage into the intended workflow.

### Preprocessing

- **Categorical features:** One-hot encoded using `OneHotEncoder(handle_unknown="ignore")`, allowing the pipeline to handle categories not seen during training.
- **Logistic Regression numerical features:** Standardized using `StandardScaler`.
- **Random Forest numerical features:** Passed through without scaling, since tree-based models generally do not require standardization.
- **Pipelines:** Preprocessing was placed inside scikit-learn pipelines so transformations are fitted on the training data and applied consistently to test data.

The numerical features used were `age`, `balance`, `day`, `campaign`, `pdays`, and `previous`. Categorical features included `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, and `poutcome`.

## Models

### 1. Logistic Regression

Logistic Regression is a baseline classification algorithm that estimates the probability of a binary outcome. It is relatively simple to interpret and provides a useful point of comparison.

The numerical features were standardized before training, while categorical features were one-hot encoded.

### 2. Random Forest Classifier

Random Forest is an ensemble model that trains multiple decision trees and combines their predictions. It can model nonlinear relationships and interactions between features.

The model was trained with 100 trees and `random_state=42`. Numerical features were passed through without scaling, and categorical features were one-hot encoded.

Both models were evaluated on the same held-out test set.

## Evaluation Results

The following are the observed test-set results from the notebook:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 69.55% | **72.32%** |
| Precision | 73.28% | **74.77%** |
| Recall | 56.24% | **62.76%** |
| F1-score | 63.64% | **68.24%** |
| ROC-AUC | 0.7579 | **0.7790** |

### Interpretation

Random Forest performed better than Logistic Regression on all five reported metrics for this split.

- **Accuracy:** Random Forest correctly classified approximately 72.32% of test records.
- **Precision:** Of the customers predicted as subscribers, approximately 74.77% actually subscribed.
- **Recall:** The model identified approximately 62.76% of the customers who actually subscribed.
- **F1-score:** Random Forest achieved a higher balance between precision and recall.
- **ROC-AUC:** A score of 0.7790 indicates useful, but imperfect, discrimination between the two target classes.

The confusion matrices also show the trade-off between missed subscribers (false negatives) and customers predicted to subscribe who did not (false positives). At the default classification threshold, Random Forest identified more actual subscribers than Logistic Regression, while also producing slightly more false positives.

Based on these results, **Random Forest was selected as the preferred model for this experiment**. This selection is based on a single train-test split and should be confirmed with more robust validation before real-world use.

## Feature Importance

Random Forest feature importance identified these as the top five features in the trained model:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Balance | 0.1514 |
| 2 | Age | 0.1313 |
| 3 | Day | 0.1170 |
| 4 | Campaign | 0.0653 |
| 5 | Pdays | 0.0429 |

Other notable features included previous campaign outcome (`poutcome_success`) and contact method categories.

These importance scores indicate how the fitted Random Forest used features when making predictions. They do **not** demonstrate that a feature causes a customer to subscribe, and importance can be affected by feature relationships and the model's training data.

## Limitations

1. **Imperfect recall:** Random Forest recall is approximately 62.76%, so it misses a meaningful portion of actual subscribers.
2. **Single train-test split:** Reported results come from one split and may vary with a different sample. Cross-validation has not been performed.
3. **No temporal validation:** The model has not been evaluated on a later time period, so its performance on future campaigns is unknown.
4. **Potential changes in customer behavior:** Campaign response patterns may change over time, reducing model performance.
5. **Feature-importance limitations:** Built-in Random Forest importance is not causal and can favor certain types of features.
6. **Business costs not quantified:** The project does not assign specific costs to false positives and false negatives or optimize the decision threshold for a defined marketing objective.

## Proposed Improvements

1. **Use cross-validation and hyperparameter tuning.** Apply stratified cross-validation and methods such as `RandomizedSearchCV` to compare settings and estimate performance more robustly. Keep a final holdout set for unbiased evaluation.
2. **Optimize the classification threshold for business needs.** Evaluate precision-recall trade-offs and select a threshold based on the relative cost of contacting unlikely subscribers versus missing likely subscribers.
3. **Evaluate on more recent data.** Where dates and suitable data are available, use a temporal holdout to test performance on a later campaign period.
4. **Improve interpretability and monitoring.** Consider permutation importance or SHAP-style explanations, and monitor model performance and data drift after deployment.

The first two items are the main proposed improvements; the additional items are further opportunities for development.

## Real-World Application

A bank could use the model to estimate which customers are more likely to subscribe to a term deposit and help a marketing team prioritize outreach. The model should support human decision-making rather than automatically exclude customers from receiving offers.

### What if a prediction is wrong?

- **False positive:** A customer is predicted to subscribe but does not. The bank may spend time or money on outreach that does not result in a subscription.
- **False negative:** A customer who would subscribe is predicted not to subscribe. The bank may miss a potential business opportunity.

The appropriate balance depends on campaign costs, customer experience, and the bank's business goals.

### Monitoring on new data

Once actual outcomes become available, the bank should periodically evaluate accuracy, precision, recall, F1-score, and ROC-AUC. It should also monitor changes in input data and customer response patterns. If performance degrades, the team should investigate the cause, validate updates, and retrain when appropriate.

### Before production use

Additional steps would include evaluation on recent unseen data, threshold selection based on business costs, bias and fairness checks, privacy and security review, reliable data pipelines, monitoring and retraining procedures, and human oversight. The model in this project is a prototype and has not been production-deployed.

## AI Tool Usage

ChatGPT was used as an assistance tool to help understand machine-learning concepts, structure the workflow, support code implementation and debugging, and interpret evaluation metrics. Code and outputs were reviewed and validated by running the notebook. The author should be prepared to explain the implementation, assumptions, results, and decisions during the follow-up discussion.

## Project Structure

A suggested structure for the submitted project is:

```text
markovate-ml-classification/
├── data/
│   └── raw/
│       └── bank.csv
├── models/
│   └── random_forest_model.pkl
├── notebooks/
│   └── 01_Markovate_Classification.ipynb
├── README.md
└── requirements.txt
```

Adjust the notebook filename and paths to match the actual project. Do not include a file in the structure unless it exists in your submission.

## How to Run the Project

### Prerequisites

- Python 3.10 or a compatible Python version
- Jupyter Notebook or JupyterLab
- The dataset file at the path expected by the notebook

### Install dependencies

Create a `requirements.txt` containing the libraries actually imported by the notebook. For the workflow described here, a starting list is:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

### Run the notebook

From the project directory, launch Jupyter:

```bash
jupyter notebook
```

Open the classification notebook and run the cells from top to bottom. Confirm that the dataset path is correct and that the notebook executes without errors.

The saved Random Forest pipeline can be loaded with `joblib` if the model file is included:

```python
import joblib

model = joblib.load("models/random_forest_model.pkl")
```

The path above assumes the code is run from the project root. If the notebook is run from its `notebooks/` directory, adjust the relative path accordingly.

---

**Note:** The reported metrics are from the current notebook run and may differ if the dataset, preprocessing, split, library versions, or model settings change. Confirm that all figures in this README match the final saved notebook before submitting.
