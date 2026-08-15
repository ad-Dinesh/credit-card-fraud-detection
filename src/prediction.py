import os
import joblib
import pandas as pd


# Get the project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Path to trained model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_card_fraud_model.pkl"
)


# Load the trained model package
model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
threshold = model_package["threshold"]
feature_columns = model_package["feature_columns"]


def predict_fraud(transaction):
    """
    Predict whether a credit card transaction is fraudulent.

    Parameters
    ----------
    transaction : pandas.DataFrame
        Transaction data containing the required model features.

    Returns
    -------
    dict
        Fraud probability, threshold, prediction and decision.
    """

    # Validate input type
    if not isinstance(transaction, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    # Check for missing features
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in transaction.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Select features in the exact training order
    transaction = transaction[feature_columns]

    # Generate fraud probability
    fraud_probability = model.predict_proba(
        transaction
    )[:, 1][0]

    # Apply optimized threshold
    prediction = int(
        fraud_probability >= threshold
    )

    # Convert prediction into readable decision
    if prediction == 1:
        decision = "Fraud"
    else:
        decision = "Legitimate"

    return {
        "fraud_probability": float(fraud_probability),
        "threshold": float(threshold),
        "prediction": prediction,
        "decision": decision
    }