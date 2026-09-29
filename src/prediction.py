import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# LOAD TRAINED MODEL ARTIFACTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_detection_model.joblib"
)

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
scaler = artifacts["scaler"]
threshold = artifacts["threshold"]
features = artifacts["features"]


# ============================================================
# BATCH FRAUD PREDICTION
# ============================================================

def predict_fraud_batch(transaction_data):
    """
    Predict fraud for multiple transactions.

    Parameters
    ----------
    transaction_data : pandas.DataFrame
        Transaction data containing all required model features.

    Returns
    -------
    predictions : list
        Predicted labels: "Fraud" or "Normal".

    fraud_probabilities : list
        Predicted fraud probabilities.
    """

    if not isinstance(transaction_data, pd.DataFrame):
        raise TypeError(
            "transaction_data must be a pandas DataFrame."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in transaction_data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    model_input = transaction_data[features].copy()

    transaction_scaled = scaler.transform(
        model_input
    )

    fraud_probabilities = model.predict_proba(
        transaction_scaled
    )[:, 1]

    predictions = [
        "Fraud"
        if probability >= threshold
        else "Normal"
        for probability in fraud_probabilities
    ]

    return predictions, fraud_probabilities