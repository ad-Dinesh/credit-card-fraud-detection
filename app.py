import streamlit as st
import pandas as pd

from src.prediction import (
    predict_fraud,
    predict_fraud_batch
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    **Machine Learning Fraud Detection System**

    Upload transaction data and use the trained
    XGBoost model to identify potentially fraudulent
    credit card transactions.
    """
)

st.divider()


# ==================================================
# MODEL INFORMATION
# ==================================================

st.subheader("Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "XGBoost"
    )

with col2:
    st.metric(
        "ROC-AUC",
        "0.9748"
    )

with col3:
    st.metric(
        "PR-AUC",
        "0.8250"
    )

with col4:
    st.metric(
        "Threshold",
        "0.3767"
    )


st.divider()


# ==================================================
# FILE UPLOAD
# ==================================================

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Upload a transaction CSV file to begin fraud detection."
    )

    st.stop()


# ==================================================
# LOAD DATA
# ==================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to read the CSV file: {error}"
    )

    st.stop()


st.success(
    f"Successfully loaded {len(data):,} transactions."
)


# ==================================================
# REQUIRED FEATURES
# ==================================================

expected_features = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


missing_features = [
    feature
    for feature in expected_features
    if feature not in data.columns
]


if missing_features:

    st.error(
        "The uploaded dataset is missing required features."
    )

    st.write(missing_features)

    st.stop()


st.success(
    "All required transaction features are available."
)


# ==================================================
# DATASET OVERVIEW
# ==================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Transactions",
        f"{len(data):,}"
    )

with col2:
    st.metric(
        "Features",
        len(expected_features)
    )

with col3:
    st.metric(
        "Dataset Size",
        f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )


# ==================================================
# PREVIEW
# ==================================================

with st.expander("View Transaction Data"):

    st.dataframe(
        data.head(20),
        use_container_width=True
    )


st.divider()


# ==================================================
# RUN BATCH PREDICTION
# ==================================================

st.subheader("🔍 Fraud Detection")

if st.button(
    "Run Fraud Detection",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Analyzing transactions..."
    ):

        try:

            results = predict_fraud_batch(
                data
            )

            st.session_state[
                "prediction_results"
            ] = results

            st.success(
                "Fraud detection completed successfully."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==================================================
# DISPLAY RESULTS
# ==================================================

if "prediction_results" in st.session_state:

    results = st.session_state[
        "prediction_results"
    ]


    # --------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------

    total_transactions = len(results)

    fraud_count = int(
        results["Prediction"].sum()
    )

    legitimate_count = (
        total_transactions - fraud_count
    )

    fraud_rate = (
        fraud_count / total_transactions * 100
        if total_transactions > 0
        else 0
    )


    # --------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "📈 Fraud Detection Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Fraud Detected",
            f"{fraud_count:,}"
        )

    with col3:

        st.metric(
            "Legitimate",
            f"{legitimate_count:,}"
        )

    with col4:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )


    # ==================================================
    # HIGH RISK TRANSACTIONS
    # ==================================================

    st.divider()

    st.subheader(
        "🚨 High-Risk Transactions"
    )

    high_risk = results[
        results["Prediction"] == 1
    ].copy()

    high_risk = high_risk.sort_values(
        by="Fraud_Probability",
        ascending=False
    )


    if len(high_risk) > 0:

        display_columns = [
            column
            for column in [
                "Time",
                "Amount",
                "Fraud_Probability",
                "Prediction",
                "Decision"
            ]
            if column in high_risk.columns
        ]

        display_data = high_risk[
            display_columns
        ].copy()

        display_data[
            "Fraud_Probability"
        ] = (
            display_data[
                "Fraud_Probability"
            ] * 100
        ).round(2)

        st.dataframe(
            display_data.head(20),
            use_container_width=True
        )

    else:

        st.success(
            "No fraudulent transactions were detected."
        )


    # ==================================================
    # FRAUD VS LEGITIMATE CHART
    # ==================================================

    st.divider()

    st.subheader(
        "Transaction Classification"
    )

    classification_counts = (
        results["Decision"]
        .value_counts()
    )

    st.bar_chart(
        classification_counts
    )


    # ==================================================
    # FRAUD PROBABILITY DISTRIBUTION
    # ==================================================

    st.subheader(
        "Fraud Probability Distribution"
    )

    probability_data = results[
        "Fraud_Probability"
    ]

    st.line_chart(
        probability_data.head(500)
    )


    # ==================================================
    # DOWNLOAD RESULTS
    # ==================================================

    st.divider()

    st.subheader(
        "⬇️ Download Results"
    )

    csv_results = results.to_csv(
        index=False
    )

    st.download_button(
        label="Download Fraud Detection Results",
        data=csv_results,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
        use_container_width=True
    )