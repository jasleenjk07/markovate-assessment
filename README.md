# Bank Marketing Classification

This project was developed as part of the Markovate assessment, **"Building an ML-Based Classification Solution."** The goal is to build a machine-learning model that predicts whether a bank customer is likely to subscribe to a term deposit.

I used Python to explore the dataset, preprocess the data, train classification models, and compare their performance.

## 1. Dataset

- **Dataset:** Bank Marketing (`bank.csv`)
- **Source:** Kaggle (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- **Records:** 11,162
- **Features:** 16 input features and 1 target column
- **Target:** `deposit` (`yes` or `no`)

The dataset contains information about customers, such as age, job, account balance, loans, and previous marketing campaign outcomes.

The target classes are reasonably balanced, with approximately 52.62% `no` and 47.38% `yes`.

## 2. Approach

I followed these steps to build the classification solution:

1. Explored the dataset and checked for missing values, duplicates, and unusual values.
2. Performed EDA to understand customer characteristics and their relationship with term deposit subscriptions.
3. Preprocessed numerical and categorical features.
4. Split the data into 80% training and 20% testing sets using stratified sampling.
5. Trained and compared Logistic Regression and Random Forest classifiers.
6. Evaluated both models using accuracy, precision, recall, F1-score, and ROC-AUC.
7. Examined feature importance and saved the trained Random Forest pipeline.

### Key preprocessing decisions

- Used one-hot encoding for categorical features.
- Standardized numerical features for Logistic Regression.
- Kept numerical features unscaled for Random Forest.
- Retained `unknown` values as categories and kept `pdays = -1` as it represents customers who were not previously contacted.
- Excluded `duration` because the goal was to predict subscription before contacting a customer. Call duration would not be available at that point.

Preprocessing was handled using scikit-learn pipelines to ensure consistent transformations during training and testing.

## 3. Models and Results

I trained two classification models and evaluated them on the same test dataset.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 69.55% | **72.32%** |
| Precision | 73.28% | **74.77%** |
| Recall | 56.24% | **62.76%** |
| F1-score | 63.64% | **68.24%** |
| ROC-AUC | 0.7579 | **0.7790** |

Random Forest performed better across all the evaluation metrics. It achieved an accuracy of 72.32% and a recall of 62.76%, making it the better-performing model in this experiment.

Based on these results, I selected Random Forest as the final model. However, further validation would be needed before using it in a real-world setting.

## 4. Feature Importance

The top five features identified by the Random Forest model were:

| Feature | Importance |
|---|---:|
| Balance | 0.1514 |
| Age | 0.1313 |
| Day | 0.1170 |
| Campaign | 0.0653 |
| Pdays | 0.0429 |

These results show which features contributed most to the model's predictions. They do not necessarily indicate a causal relationship with customer subscriptions.

## 5. Limitations and Future Improvements

Although Random Forest performed better than Logistic Regression, there is still room for improvement.

Some possible improvements include:

- Using cross-validation and hyperparameter tuning to improve model reliability.
- Adjusting the classification threshold based on the bank's marketing goals.
- Testing the model on more recent data.
- Using additional techniques to understand feature importance and monitor model performance.

The current model is a prototype and has not been deployed in a production environment.

## 6. Real-World Application

A bank could use this model to identify customers who are more likely to subscribe to a term deposit and prioritize its marketing efforts.

However, the predictions should be used to support decisions rather than automatically exclude customers from receiving offers. The model would also need further testing and monitoring before practical use.

## 7. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## 8. Project Structure

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

Update the filenames and folders to match your actual submission.

## 9. How to Run

**1. Clone or download the project** and navigate to its directory.

**2. Install the required libraries:**

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file should include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

**3. Open Jupyter Notebook:**

```bash
jupyter notebook
```

**4.** Open the classification notebook and run the cells in order. Make sure the dataset path is correct.

If the saved model is included, load it using:

```python
import joblib

model = joblib.load("models/random_forest_model.pkl")
```