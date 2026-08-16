import os
import joblib
import pandas as pd


# --------------------------------------------------
# Model Path
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_card_fraud_model.pkl"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
threshold = model_package["threshold"]
feature_columns = model_package["feature_columns"]


# --------------------------------------------------
# Validate Features
# --------------------------------------------------

def validate_features(transaction):
    """
    Validate that all required model features
    are present in the input DataFrame.
    """

    if not isinstance(transaction, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in transaction.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )


# --------------------------------------------------
# Single Transaction Prediction
# --------------------------------------------------

def predict_fraud(transaction):
    """
    Predict fraud for a single transaction.
    """

    validate_features(transaction)

    transaction = transaction[
        feature_columns
    ]

    fraud_probability = model.predict_proba(
        transaction
    )[:, 1][0]

    prediction = int(
        fraud_probability >= threshold
    )

    decision = (
        "Fraud"
        if prediction == 1
        else "Legitimate"
    )

    return {
        "fraud_probability": float(
            fraud_probability
        ),
        "threshold": float(threshold),
        "prediction": prediction,
        "decision": decision
    }


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

def predict_fraud_batch(transactions):
    """
    Predict fraud for multiple transactions.
    """

    validate_features(transactions)

    model_input = transactions[
        feature_columns
    ]

    fraud_probabilities = model.predict_proba(
        model_input
    )[:, 1]

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    decisions = [
        "Fraud" if prediction == 1
        else "Legitimate"
        for prediction in predictions
    ]

    results = transactions.copy()

    results["Fraud_Probability"] = (
        fraud_probabilities
    )

    results["Prediction"] = predictions

    results["Decision"] = decisions

    return results