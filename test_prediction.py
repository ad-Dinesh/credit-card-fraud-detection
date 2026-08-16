import pandas as pd

from src.prediction import predict_fraud_batch


# --------------------------------------------------
# Load Original Dataset
# --------------------------------------------------

DATA_PATH = "data/raw/creditcard.csv"

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Select Sample Transactions
# --------------------------------------------------

sample_data = df.head(10)


# --------------------------------------------------
# Run Batch Prediction
# --------------------------------------------------

results = predict_fraud_batch(
    sample_data
)


# --------------------------------------------------
# Display Results
# --------------------------------------------------

print(results[
    [
        "Fraud_Probability",
        "Prediction",
        "Decision"
    ]
])