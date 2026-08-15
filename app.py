import streamlit as st
import pandas as pd

from src.prediction import predict_fraud


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    ### Machine Learning Fraud Detection System

    Upload a transaction CSV file and the trained XGBoost
    model will evaluate the transactions for potential fraud.
    """
)

st.divider()


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

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

st.divider()


# --------------------------------------------------
# File Upload
# --------------------------------------------------

st.subheader("Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Read uploaded CSV
        data = pd.read_csv(uploaded_file)

        st.success(
            f"File uploaded successfully: {len(data):,} transactions"
        )

        # --------------------------------------------------
        # Dataset Preview
        # --------------------------------------------------

        st.subheader("Transaction Data Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # --------------------------------------------------
        # Required Features
        # --------------------------------------------------

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
                "The uploaded file is missing required features:"
            )

            st.write(missing_features)

        else:

            st.success(
                "All required transaction features are available."
            )

            # --------------------------------------------------
            # Select Transaction
            # --------------------------------------------------

            st.subheader("Transaction Selection")

            transaction_index = st.number_input(
                "Select transaction index",
                min_value=0,
                max_value=len(data) - 1,
                value=0,
                step=1
            )

            selected_transaction = data.iloc[
                [transaction_index]
            ]

            st.write("Selected Transaction")

            st.dataframe(
                selected_transaction,
                use_container_width=True
            )

            # --------------------------------------------------
            # Prediction
            # --------------------------------------------------

            if st.button(
                "🔍 Analyze Transaction",
                type="primary"
            ):

                try:

                    result = predict_fraud(
                        selected_transaction
                    )

                    fraud_probability = (
                        result["fraud_probability"] * 100
                    )

                    decision = result["decision"]

                    st.divider()

                    st.subheader(
                        "Fraud Detection Result"
                    )

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        st.metric(
                            "Fraud Probability",
                            f"{fraud_probability:.2f}%"
                        )

                    with result_col2:

                        st.metric(
                            "Decision",
                            decision
                        )

                    # --------------------------------------------------
                    # Decision Message
                    # --------------------------------------------------

                    if decision == "Fraud":

                        st.error(
                            "⚠️ Potential fraudulent transaction detected."
                        )

                    else:

                        st.success(
                            "✅ Transaction classified as legitimate."
                        )

                    # --------------------------------------------------
                    # Threshold
                    # --------------------------------------------------

                    st.caption(
                        f"Decision threshold: "
                        f"{result['threshold']:.4f}"
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

    except Exception as error:

        st.error(
            f"Unable to read the uploaded file: {error}"
        )


else:

    st.info(
        "Please upload a CSV file containing transaction data."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning Project | XGBoost"
)