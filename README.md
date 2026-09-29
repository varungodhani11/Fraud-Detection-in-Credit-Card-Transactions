# Fraud Detection in Credit Card Transactions

A machine learning project for detecting potentially fraudulent credit card transactions using supervised and unsupervised learning techniques.

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small proportion of all transactions. This project develops and evaluates multiple machine learning approaches for identifying potentially fraudulent transactions.

The project includes exploratory data analysis, preprocessing, feature scaling, class-imbalance handling, supervised model comparison, unsupervised anomaly detection, threshold tuning, feature importance analysis, and a Streamlit application for batch transaction prediction.

## Objectives

- Analyze the distribution and characteristics of fraudulent transactions.
- Prepare the transaction data for machine learning.
- Handle the severe class imbalance in the target variable.
- Train and evaluate multiple supervised classification models.
- Compare supervised models using fraud-focused evaluation metrics.
- Explore unsupervised anomaly detection using Isolation Forest.
- Tune classification thresholds using validation data.
- Identify important predictive features.
- Save the trained model and preprocessing artifacts.
- Provide a Streamlit interface for batch fraud prediction.

## Dataset

The project uses the Credit Card Fraud Detection dataset containing transactions made by European cardholders.

### Dataset Source

The dataset was obtained from Kaggle:

https://www.kaggle.com/mlg-ulb/creditcardfraud

### Dataset Setup

To run the notebook locally:

1. Download the dataset from Kaggle.
2. Extract the downloaded files.
3. Place `creditcard.csv` inside the project's `data/` folder.

Expected local path:

```text
data/creditcard.csv
```

The dataset contains:

- `284,807` original transactions
- `30` input features
- `1` target column: `Class`
- `492` fraud transactions in the original dataset

### Features

The transaction data contains:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class` — target variable

The `V1` to `V28` variables are anonymized numerical features.

For the final modeling workflow, duplicate rows were removed before training.

After duplicate removal:

- Rows: `283,726`
- Normal transactions: `283,253`
- Fraudulent transactions: `473`

The final model uses the 30 input features:

`Time`, `V1`–`V28`, and `Amount`.

## Machine Learning Workflow

### 1. Exploratory Data Analysis

The notebook analyzes:

- Target-class distribution
- Transaction amount distribution
- Transaction time patterns
- Feature statistics
- Feature differences between normal and fraudulent transactions
- Feature correlations with the target
- Feature distribution visualizations

Because the dataset is highly imbalanced, accuracy alone is not treated as the primary indicator of fraud-detection performance.

### 2. Data Preprocessing

The preprocessing workflow includes:

- Duplicate removal
- Separation of features and target
- Stratified train/test split
- Standardization using `StandardScaler`
- Class-imbalance handling for model training
- Validation split for threshold selection

The train/test split uses an 80/20 stratified split with `random_state=42`.

### 3. Supervised Models

The following supervised models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Classification thresholds were tuned using the validation set rather than directly optimizing on the final test set.

### 4. Unsupervised Model

Isolation Forest was evaluated as an anomaly-detection approach without using the fraud labels during model training.

### 5. Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

For this highly imbalanced problem, Precision, Recall, F1-score, and PR-AUC are particularly important.

## Final Test Results

The following results were obtained on the held-out test set.

| Model | Threshold | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.99 | 0.9989 | 0.6250 | 0.7895 | 0.6977 | 0.9617 | 0.6886 |
| Random Forest | 0.38 | 0.9995 | 0.9231 | 0.7579 | 0.8324 | 0.9497 | 0.8133 |
| XGBoost | 0.79 | 0.9995 | 0.9726 | 0.7474 | 0.8452 | 0.9707 | 0.8153 |
| Isolation Forest | 0.1534 | 0.9974 | 0.2283 | 0.2211 | 0.2246 | 0.9308 | 0.1143 |

The final application uses the trained **XGBoost** model with the validation-selected classification threshold of approximately `0.79`.

## Feature Importance

Feature importance analysis was performed using the trained XGBoost and Random Forest models.

The most prominent features included:

- `V14`
- `V4`
- `V10`
- `V12`
- `V17`
- `V3`
- `V11`
- `V16`

Feature importance indicates model contribution and should not be interpreted as proof of causal relationships.

## Prediction Pipeline

The saved model artifact contains:

- Trained XGBoost model
- Fitted StandardScaler
- Selected classification threshold
- Required feature names

The application uses these artifacts to:

1. Read uploaded transaction data.
2. Validate the required model features.
3. Select the required 30 features.
4. Apply the saved scaler.
5. Generate fraud probabilities.
6. Apply the selected classification threshold.
7. Return `Fraud` or `Normal` predictions.
8. Display and download prediction results.

## Streamlit Application

The project includes a Streamlit application for batch transaction prediction.

### Input Requirements

The uploaded CSV must contain the same 30 model input features:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
```

The `Class` column is not required for prediction.

The application can accept additional columns, but only the required model features are passed to the trained model.

### Application Features

- CSV upload
- Dataset preview
- Dataset statistics
- Required-feature validation
- Batch fraud prediction
- Fraud probability
- Detection summary
- Prediction results table
- CSV download of predictions
- Model and threshold information

## Project Structure

```text
Fraud-Detection-in-Credit-Card-Transactions/
│
├── data/
│   └── creditcard.csv
│
├── notebook/
│   └── Fraud_Detection_in_Credit_Card_Transactions.ipynb
│
├── models/
│   └── fraud_detection_model.joblib
│
├── src/
│   └── prediction.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

> Note: The original `creditcard.csv` dataset is not included in the GitHub repository. Download it separately from the Kaggle source and place it in the `data/` directory as described above.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/varungodhani11/Fraud-Detection-in-Credit-Card-Transactions.git
cd Fraud-Detection-in-Credit-Card-Transactions
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it at:

```text
data/creditcard.csv
```

## Run the Streamlit Application

From the project root directory:

```bash
streamlit run app.py
```

The application will open in the browser.

The Streamlit application does not require the original `creditcard.csv` dataset for prediction because it uses the saved trained model artifact.

## Model Artifact

The trained model and preprocessing components are stored in:

```text
models/fraud_detection_model.joblib
```

The artifact contains:

- Trained XGBoost model
- Fitted scaler
- Selected classification threshold
- Required feature names

The artifact is loaded by:

```text
src/prediction.py
```

## Notebook

The complete machine learning workflow is available in:

```text
notebook/Fraud_Detection_in_Credit_Card_Transactions.ipynb
```

The notebook covers:

- Exploratory data analysis
- Data preprocessing
- Model training
- Validation
- Threshold tuning
- Model evaluation
- Confusion matrices
- ROC and precision-recall analysis
- Feature importance
- Single-transaction prediction testing
- Model artifact creation

## Important Notes

- This project is intended for educational and analytical purposes.
- The dataset is highly imbalanced, so accuracy should not be interpreted in isolation.
- A fraud prediction is a model output and should not be treated as a definitive determination of fraud.
- The application expects transaction features compatible with the training data.
- The trained model was evaluated on a held-out test set.
- The original dataset is excluded from version control through `.gitignore`.

## Future Improvements

Potential extensions include:

- Additional threshold optimization based on business costs.
- Probability calibration.
- More extensive hyperparameter tuning.
- Cross-validation for model selection.
- Explainable AI techniques such as SHAP.
- More robust input validation in the Streamlit application.
- Real-time transaction-stream integration.
- Model monitoring and drift detection.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
