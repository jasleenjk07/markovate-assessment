# Bank Marketing Classification

This is my machine-learning project for the Markovate assessment, “Building an ML-Based Classification Solution.” I used the Bank Marketing dataset to explore whether customer and campaign information can help predict if someone will subscribe to a term deposit.

The project is built around one Jupyter notebook. It covers data exploration, preprocessing, training two classifiers, comparing their results, and saving the Random Forest pipeline.

## Project goal

The goal is to predict the `deposit` outcome:

- `no` means the customer did not subscribe.
- `yes` means the customer subscribed.

I treated this as a prediction made before contacting a customer. For that reason, I left out `duration`: the length of a call is not known before the call and could give the model information that would not be available when making the prediction.

## Dataset

Dataset link:- https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

The notebook uses a file named `bank.csv` at `data/raw/bank.csv`. The dataset has 11,162 rows and 17 columns, including the `deposit` target. In the copy used for this project, there are 5,873 `no` records and 5,289 `yes` records.

Some categorical columns contain the value `unknown`. I kept these as categories rather than treating them as missing values. The `pdays` value of `-1`, which represents a customer who was not previously contacted, was also kept as-is.

## What I did

I started by checking the dataset's shape, data types, missing values, duplicates, category values, and numerical summaries. I also explored the target distribution and compared subscription rates across age groups, jobs, balance groups, contact months, and previous campaign outcomes.

For modelling, I split the data into training and test sets using an 80:20 split, stratified by the target, with `random_state=42`. The notebook uses six numerical features (`age`, `balance`, `day`, `campaign`, `pdays`, and `previous`) and nine categorical features.

I used separate preprocessing pipelines for the two models:

- **Logistic Regression:** numerical features are standardized and categorical features are one-hot encoded.
- **Random Forest:** numerical features are passed through without scaling and categorical features are one-hot encoded.

Both models use the same train-test split. The preprocessing is part of each scikit-learn pipeline, so it is fitted on the training data and then applied to the test data.

## Results

These are the results recorded in the notebook's model comparison table for the held-out test set:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 69.55% | **72.32%** |
| Precision | 73.28% | **74.77%** |
| Recall | 56.24% | **62.76%** |
| F1-score | 63.64% | **68.24%** |
| ROC-AUC | 0.7579 | **0.7790** |

Random Forest scored higher on all five metrics in this run, so I selected it as the preferred model for this experiment. Its recall was 62.76%, which means it still missed some customers who subscribed. Whether that is acceptable would depend on the campaign's goals and the cost of contacting customers who do not subscribe.

The notebook also plots confusion matrices and ranks features by the fitted Random Forest's importance scores. Balance, age, and day were the top three features in that ranking. These scores describe how the model used the features; they do not show that those features cause a customer to subscribe.

The results come from one train-test split, so they should be treated as an initial comparison rather than a guarantee of future performance.

## Limitations and next steps

This is a learning project, not a production-ready system. I have not used cross-validation, tested the model on data from a later campaign, or selected a classification threshold based on business costs. Customer behaviour and campaign results may also change over time.

If I continued the project, I would:

1. Use stratified cross-validation and tune the model while keeping a final test set for evaluation.
2. Compare different prediction thresholds based on the cost of false positives and false negatives.
3. Test the model on more recent data, if it is available.
4. Check for bias, monitor performance over time, and review changes before retraining or using the model in practice.

## Project files

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   └── Customer_Subscription_Classification.ipynb
├── data/
│   └── raw/
│       └── bank.csv                 # add locally; excluded from Git
└── models/
    └── random_forest_model.pkl      # created by the notebook; excluded from Git
```

The notebook is the only analysis and modelling code in the repository. The `src/` and `results/` directories do not currently contain project files. The dataset and saved model are excluded by `.gitignore`, so they will not be present when the repository is cloned.

## Running the notebook

You will need Python and Jupyter. The project dependencies are listed in `requirements.txt`.

1. Clone the repository and move into the project directory.
2. Create and activate a virtual environment if you want to keep the project dependencies separate:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with `.venv\Scripts\activate`.

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Put the dataset at `data/raw/bank.csv`.
5. Before running the notebook, update its first data-loading cell. It currently contains an absolute path from my computer; replace it with a path that works on yours, for example:

   ```python
   df = pd.read_csv("../data/raw/bank.csv")
   ```

   This relative path assumes the notebook is running with `notebooks/` as its working directory.

6. Open `notebooks/Customer_Subscription_Classification.ipynb` in Jupyter and run the cells from top to bottom. The notebook saves the Random Forest pipeline to `models/random_forest_model.pkl` and checks that the reloaded model gives the same test predictions.

The saved model is also excluded by `.gitignore`; it is created locally when the relevant notebook cells run.
