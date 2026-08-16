import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

from src.prediction import (
    model,
    predict_fraud,
    predict_fraud_batch,
    threshold
)

# ==================================================
# MODEL METADATA / THRESHOLD
# ==================================================

try:
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    metadata = {}

DEFAULT_THRESHOLD = float(metadata.get("threshold", 0.3767))
selected_threshold = float(threshold if threshold is not None else DEFAULT_THRESHOLD)

EXPECTED_FEATURES = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)

THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.3767,
                   0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ==================================================
# HELPERS
# ==================================================

def load_csv(uploader_key, label):
    """Upload + read a CSV. Returns a DataFrame or None."""
    uploaded_file = st.file_uploader(label, type=["csv"], key=uploader_key)
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        return None


def check_features(df):
    """Verify the uploaded dataset has every column the model expects."""
    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing:
        st.error("The dataset is missing required model features.")
        st.write(missing)
        return False
    return True


def compute_business_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    missed_fraud_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    fraud_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return tn, fp, fn, tp, fraud_detection_rate, false_alarm_rate, missed_fraud_rate, fraud_precision


def render_threshold_analysis(y_true, y_prob):
    st.subheader("Threshold Analysis")

    rows = []
    for t in THRESHOLD_GRID:
        preds_t = (y_prob >= t).astype(int)
        rep_t = classification_report(y_true, preds_t, output_dict=True, zero_division=0)
        rows.append({
            "Threshold": t,
            "Precision": rep_t["1"]["precision"],
            "Recall": rep_t["1"]["recall"],
            "F1-Score": rep_t["1"]["f1-score"]
        })
    threshold_df = pd.DataFrame(rows)

    st.dataframe(
        threshold_df.style.format({
            "Threshold": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1-Score": "{:.4f}"
        }),
        use_container_width=True
    )

    fig, ax = plt.subplots()
    ax.plot(threshold_df["Threshold"], threshold_df["Precision"], marker="o", label="Precision")
    ax.plot(threshold_df["Threshold"], threshold_df["Recall"], marker="o", label="Recall")
    ax.plot(threshold_df["Threshold"], threshold_df["F1-Score"], marker="o", label="F1-Score")
    ax.axvline(selected_threshold, linestyle="--", label=f"Selected Threshold = {selected_threshold:.4f}")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance vs Classification Threshold")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("Credit Card Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Transaction Analysis",
        "📊 Model Performance",
        "💼 Business Impact",
        "🧠 Explainability"
    ]
)

# ==================================================
# 🏠 DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.title("💳 Credit Card Fraud Detection")
    st.caption(
        "End-to-end Machine Learning system for detecting "
        "fraudulent credit card transactions using XGBoost."
    )
    st.divider()

    st.subheader("Model Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", "XGBoost")
    with col2:
        st.metric("ROC-AUC", "0.9748")
    with col3:
        st.metric("PR-AUC", "0.8250")
    with col4:
        st.metric("Threshold", f"{selected_threshold:.4f}")

    st.divider()

    st.subheader("🎯 Project Objective")
    st.write(
        """
        The objective of this project is to detect fraudulent
        credit card transactions using machine learning while
        handling severe class imbalance and optimizing the
        classification threshold for fraud detection.
        """
    )

    st.subheader("🔄 Machine Learning Pipeline")
    pipeline = [
        "Data Collection", "Exploratory Data Analysis", "Data Preprocessing",
        "Train/Test Split", "Class Imbalance Analysis", "Model Training",
        "XGBoost Optimization", "Threshold Optimization",
        "SHAP Explainability", "Deployment"
    ]
    for i, step in enumerate(pipeline, start=1):
        st.write(f"**{i}.** {step}")

    st.divider()

    st.subheader("🔐 System Characteristics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Imbalanced Dataset**\n\nFraud represents a very small proportion of total transactions.")
    with col2:
        st.info("**Optimized Threshold**\n\nThe decision threshold is optimized instead of relying blindly on 0.50.")
    with col3:
        st.info("**Explainable AI**\n\nSHAP is used to understand which features influence predictions.")


# ==================================================
# 🔍 TRANSACTION ANALYSIS
# ==================================================

elif page == "🔍 Transaction Analysis":

    st.header("🔍 Transaction Analysis")
    st.write(
        "Upload a transaction dataset to identify potentially "
        "fraudulent transactions using the trained XGBoost model."
    )

    data = load_csv("transaction_upload", "Upload transaction CSV")

    if data is not None and check_features(data):

        st.success(f"Successfully loaded {len(data):,} transactions.")

        with st.expander("View Transaction Data"):
            st.dataframe(data.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Transactions", f"{len(data):,}")
        with col2:
            st.metric("Model Features", len(EXPECTED_FEATURES))

        st.divider()

        if st.button("🔍 Run Fraud Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing transactions..."):
                try:
                    st.session_state["transaction_results"] = predict_fraud_batch(data)
                    st.success("Fraud detection completed successfully.")
                except Exception as error:
                    st.error(f"Prediction failed: {error}")

        if "transaction_results" in st.session_state:

            results = st.session_state["transaction_results"]

            st.divider()
            st.subheader("📊 Detection Summary")

            total = len(results)
            fraud_count = int(results["Prediction"].sum())
            legit_count = total - fraud_count
            fraud_rate = (fraud_count / total * 100) if total > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", f"{total:,}")
            with col2:
                st.metric("Fraud", f"{fraud_count:,}")
            with col3:
                st.metric("Legitimate", f"{legit_count:,}")
            with col4:
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            st.subheader("🚨 High-Risk Transactions")
            high_risk = results[results["Prediction"] == 1].sort_values(
                "Fraud_Probability", ascending=False
            )

            if len(high_risk) > 0:
                display_columns = [
                    c for c in ["Time", "Amount", "Fraud_Probability", "Prediction", "Decision"]
                    if c in high_risk.columns
                ]
                display_data = high_risk[display_columns].copy()
                display_data["Fraud_Probability"] = (display_data["Fraud_Probability"] * 100).round(2)
                st.dataframe(display_data.head(20), use_container_width=True)
            else:
                st.success("No fraudulent transactions detected.")

            st.subheader("Transaction Classification")
            st.bar_chart(results["Decision"].value_counts())

            st.subheader("Fraud Probability Distribution")
            st.line_chart(results["Fraud_Probability"].head(500))

            st.subheader("⬇️ Download Results")
            st.download_button(
                label="Download Fraud Detection Results",
                data=results.to_csv(index=False),
                file_name="fraud_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==================================================
# 📊 MODEL PERFORMANCE
# ==================================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")
    st.write("Evaluate the XGBoost fraud detection model using ground-truth transaction labels.")

    data = load_csv("performance_upload", "Upload labeled transaction CSV")

    if data is not None:

        if "Class" not in data.columns:
            st.error("Model evaluation requires a 'Class' column.")
            st.info("Use the original labeled creditcard.csv dataset for model evaluation.")
            st.stop()

        if not check_features(data):
            st.stop()

        try:
            results = predict_fraud_batch(data)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

        y_true = data["Class"]
        y_pred = results["Prediction"]
        y_prob = results["Fraud_Probability"]

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        f1_score = report["1"]["f1-score"]
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        st.subheader("Model Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Precision", f"{precision:.4f}")
        with col2: st.metric("Recall", f"{recall:.4f}")
        with col3: st.metric("F1-Score", f"{f1_score:.4f}")
        with col4: st.metric("ROC-AUC", f"{roc_auc:.4f}")
        with col5: st.metric("PR-AUC", f"{pr_auc:.4f}")

        st.divider()
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic")
        ax_roc.legend()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

        st.subheader("Precision-Recall Curve")
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall_curve, precision_curve, label=f"XGBoost (AP = {pr_auc:.4f})")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.legend()
        st.pyplot(fig_pr)
        plt.close(fig_pr)

        st.divider()
        render_threshold_analysis(y_true, y_prob)


# ==================================================
# 💼 BUSINESS IMPACT
# ==================================================

elif page == "💼 Business Impact":

    st.header("💼 Business Impact Analysis")
    st.write("This section translates model predictions into business-oriented fraud detection metrics.")

    data = load_csv("business_upload", "Upload labeled transaction CSV")

    if data is not None:

        if "Class" not in data.columns:
            st.error("Business impact analysis requires the 'Class' column.")
            st.stop()

        if not check_features(data):
            st.stop()

        try:
            results = predict_fraud_batch(data)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

        y_true = data["Class"]
        y_pred = results["Prediction"]

        (tn, fp, fn, tp, fraud_detection_rate,
         false_alarm_rate, missed_fraud_rate, fraud_precision) = compute_business_metrics(y_true, y_pred)

        st.subheader("Fraud Detection KPIs")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Fraud Detection Rate", f"{fraud_detection_rate * 100:.2f}%")
        with col2: st.metric("False Alarm Rate", f"{false_alarm_rate * 100:.2f}%")
        with col3: st.metric("Missed Fraud Rate", f"{missed_fraud_rate * 100:.2f}%")
        with col4: st.metric("Fraud Precision", f"{fraud_precision * 100:.2f}%")

        st.divider()
        st.subheader("Fraud Detection Outcomes")
        outcome_data = pd.DataFrame({
            "Outcome": ["True Negatives", "False Positives", "False Negatives", "True Positives"],
            "Transactions": [int(tn), int(fp), int(fn), int(tp)]
        })
        st.dataframe(outcome_data, use_container_width=True, hide_index=True)
        st.bar_chart(outcome_data.set_index("Outcome"))

        if "Amount" in results.columns and "Class" in results.columns:
            st.divider()
            st.subheader("💰 Transaction Amount Analysis")

            actual_fraud = results[results["Class"] == 1]
            detected_fraud = results[(results["Class"] == 1) & (results["Prediction"] == 1)]
            missed_fraud = results[(results["Class"] == 1) & (results["Prediction"] == 0)]

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Actual Fraud Amount", f"${actual_fraud['Amount'].sum():,.2f}")
            with col2: st.metric("Detected Fraud Amount", f"${detected_fraud['Amount'].sum():,.2f}")
            with col3: st.metric("Missed Fraud Amount", f"${missed_fraud['Amount'].sum():,.2f}")

            st.caption(
                "These values represent transaction amounts in the dataset "
                "and should not be interpreted as confirmed financial losses or savings."
            )

        st.divider()
        st.subheader("📌 Business Interpretation")
        st.markdown(
            f"""
            - **True Positives ({int(tp)})**: fraudulent transactions correctly detected by the model.
            - **False Positives ({int(fp)})**: legitimate transactions incorrectly flagged for investigation.
            - **False Negatives ({int(fn)})**: fraudulent transactions missed by the model.
            - **True Negatives ({int(tn)})**: legitimate transactions correctly accepted.

            The current model detects approximately **{fraud_detection_rate * 100:.2f}%**
            of the fraudulent transactions in the evaluated dataset.

            The false alarm rate is approximately **{false_alarm_rate * 100:.2f}%**, meaning
            only a small proportion of legitimate transactions are incorrectly flagged.
            """
        )


# ==================================================
# 🧠 EXPLAINABILITY
# ==================================================

elif page == "🧠 Explainability":

    st.header("🧠 Model Explainability")
    st.write(
        """
        SHAP (SHapley Additive exPlanations) is used to
        understand how individual features influence the
        fraud prediction.
        """
    )
    st.info(
        "Upload the original labeled transaction dataset "
        "or an unlabeled transaction dataset containing "
        "the required model features."
    )

    data = load_csv("explainability_upload", "Upload transaction CSV")

    if data is not None and check_features(data):

        st.subheader("Select Transaction")
        transaction_index = st.number_input(
            "Transaction index", min_value=0, max_value=len(data) - 1, value=0, step=1
        )
        selected_transaction = data.loc[[transaction_index], EXPECTED_FEATURES]

        prediction_result = predict_fraud(selected_transaction)
        fraud_probability = prediction_result["fraud_probability"]
        decision = prediction_result["decision"]

        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Fraud Probability", f"{fraud_probability * 100:.2f}%")
        with col2: st.metric("Decision", decision)
        with col3: st.metric("Threshold", f"{selected_threshold:.4f}")

        st.divider()
        st.subheader("Feature Contributions")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(selected_transaction)

            if isinstance(shap_values, list):
                shap_row = shap_values[1][0]
            else:
                shap_row = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]

            contribution_data = pd.DataFrame({
                "Feature": EXPECTED_FEATURES,
                "SHAP_Value": shap_row,
                "Absolute_Impact": abs(shap_row)
            }).sort_values("Absolute_Impact", ascending=False).head(10)

            display_contributions = contribution_data[["Feature", "SHAP_Value"]].copy()
            display_contributions["SHAP_Value"] = display_contributions["SHAP_Value"].round(6)
            st.dataframe(display_contributions, use_container_width=True, hide_index=True)

            fig_shap, ax_shap = plt.subplots()
            shap_plot_data = contribution_data.sort_values("SHAP_Value")
            ax_shap.barh(shap_plot_data["Feature"], shap_plot_data["SHAP_Value"])
            ax_shap.axvline(0, linestyle="--")
            ax_shap.set_xlabel("SHAP Value")
            ax_shap.set_ylabel("Feature")
            ax_shap.set_title("Top Feature Contributions")
            st.pyplot(fig_shap)
            plt.close(fig_shap)

            st.subheader("How to Interpret This")
            st.markdown(
                """
                **Positive SHAP values** push the prediction toward the fraud class.

                **Negative SHAP values** push the prediction toward the legitimate class.

                Larger absolute SHAP values indicate stronger influence on the individual prediction.
                """
            )

        except Exception as error:
            st.error(f"SHAP explanation could not be generated: {error}")