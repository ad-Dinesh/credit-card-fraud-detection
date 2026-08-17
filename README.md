# 💳 Credit Card Fraud Detection

### An end-to-end machine learning system for detecting fraudulent credit card transactions using XGBoost, SHAP, and Streamlit

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model Results](#-model-results)
- [Business Impact Analysis](#-business-impact-analysis)
- [Explainability (SHAP)](#-explainability-shap)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Testing](#-testing)
- [Tech Stack](#️-tech-stack)
- [Limitations](#️-limitations)
- [Roadmap](#-roadmap)
- [Resume / Interview Notes](#-resume--interview-notes)
- [Data Privacy & Security](#-data-privacy--security)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔎 Overview

Credit card fraud makes up a tiny fraction of a percent of all transactions, which is what makes fraud detection a genuinely hard machine learning problem: a model that predicts "legitimate" 100% of the time can still look 99.8% accurate while catching zero fraud.

This project implements a complete, reproducible pipeline:

- EDA and class-imbalance analysis
- A tuned **XGBoost** classifier benchmarked against baseline models
- **Threshold optimization** instead of a naive 0.5 cutoff
- **SHAP-based explainability** for individual predictions
- **Business-impact translation** of model metrics
- A multi-page **Streamlit application** for batch scoring and analysis

---

## 🎯 Problem Statement

Given an anonymized credit card transaction, predict whether it is **fraudulent** or **legitimate**, while accounting for:

1. **Extreme class imbalance** (fraud is ~0.17% of transactions)
2. **Asymmetric costs** — a missed fraud (false negative) is typically far more expensive than a false alarm (false positive)
3. **Interpretability** — analysts need to know *why* a transaction was flagged

---

## ✨ Key Features

- Exploratory data analysis with fraud vs. legitimate behavior comparison
- Class-imbalance handling via SMOTE experimentation
- XGBoost classifier tuned for imbalanced tabular data
- Data-driven threshold optimization (`0.3767`) instead of the default `0.5`
- SHAP explainability — global and per-transaction feature attribution
- Business-impact reporting (detection rate, false-alarm rate, financial exposure)
- Interactive Streamlit app: dashboard, batch prediction, performance, business impact, explainability
- Downloadable scored results (CSV)
- Automated tests for the prediction pipeline
- Reproducible environment via pinned dependencies

---

## 🏗️ Architecture

```text
Raw Transaction Data (creditcard.csv)
            │
            ▼
   EDA & Preprocessing
   (scaling, imbalance analysis)
            │
            ▼
   Model Experiments
   (baselines + XGBoost + SMOTE)
            │
            ▼
   Threshold Tuning (PR curve)
            │
            ▼
   Saved Model Artifact (.pkl + metadata)
            │
   ┌────────┼────────┐
   ▼        ▼         ▼
Prediction  SHAP    Evaluation +
 Engine   Explainer  Business Impact
   │        │         │
   └────────┼─────────┘
            ▼
    Streamlit Dashboard
```

---

## 📂 Repository Structure

```text
credit-card-fraud-detection/
│
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Pinned dependencies
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│   └── creditcard.csv              # Not tracked in Git
│
├── models/
│   ├── credit_card_fraud_model.pkl # Trained XGBoost model
│   └── model_metadata.json         # Threshold, feature list, training metrics
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_xgboost_training.ipynb
│   ├── 05_threshold_optimization.ipynb
│   └── 06_shap_explainability.ipynb
│
├── src/
│   └── prediction.py                # Reusable inference logic
│
├── tests/
│   └── test_prediction.py           # Unit tests
│
└── docs/
    └── architecture.md
```

---

## 🗃️ Dataset

This project uses the **[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** (European cardholder transactions, released by the Machine Learning Group at ULB).

| Property | Detail |
|---|---|
| Transactions | 284,807 |
| Fraudulent transactions | 492 (~0.172%) |
| Features | `Time`, `Amount`, and 28 PCA-transformed features `V1`–`V28` |
| Target | `Class` (1 = fraud, 0 = legitimate) |

To use your own copy of the data, place `creditcard.csv` inside the `data/` directory.

---

## 🔬 Machine Learning Workflow

**1. Data Understanding & EDA** — shape, dtypes, missing values, class distribution, transaction amount patterns.

**2. Handling Class Imbalance** — since fraud is extremely rare, accuracy is intentionally not the primary metric. The project evaluates precision, recall, F1-score, ROC-AUC, and PR-AUC instead, and tests SMOTE as an experimental resampling strategy.

**3. Model Development** — baseline models (logistic regression, random forest) were benchmarked before selecting **XGBoost**, based on PR-AUC performance and native handling of imbalance via `scale_pos_weight`.

**4. Threshold Optimization** — the decision threshold was selected from the precision-recall curve rather than defaulting to 0.50:

```text
Selected threshold: 0.3767
```

**5. Explainability** — SHAP values are computed to understand which features drive each individual prediction.

---

## 📊 Model Results

| Metric | Value |
|---|---:|
| Model | XGBoost |
| ROC-AUC | **0.9747** |
| PR-AUC | **0.8250** |
| Selected Threshold | **0.3767** |

> These numbers reflect the current evaluation setup on held-out data and are not a guarantee of production performance.

---

## 💼 Business Impact Analysis

The app translates model output into terms a fraud/risk team would care about:

- True Positives / True Negatives / False Positives / False Negatives
- Fraud detection rate
- False alarm rate
- Missed fraud rate
- Fraud precision
- Actual, detected, and missed fraud transaction amounts

---

## 🧠 Explainability (SHAP)

```text
Positive SHAP value  →  pushes prediction toward FRAUD
Negative SHAP value  →  pushes prediction toward LEGITIMATE
```

A larger absolute SHAP value means that feature had a stronger influence on that specific prediction. The Streamlit app lets you pick an individual transaction and inspect its full feature-contribution breakdown.

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/<YOUR_USERNAME>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Create a virtual environment**

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

**Run the Streamlit app**
```bash
streamlit run app.py
```
Then open the URL shown in your terminal (typically `http://localhost:8501`).

**Use the prediction module directly**
```python
from src.prediction import predict_transaction

result = predict_transaction(transaction_features)
print(result["fraud_probability"], result["is_fraud"])
```

**Batch scoring via the app**
1. Go to **Transaction Analysis**
2. Upload a CSV of transactions
3. Review flagged high-risk transactions
4. Download the scored results

---

## 🧪 Testing

```bash
python tests/test_prediction.py
python -m py_compile app.py
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | Pandas, NumPy |
| Modeling | XGBoost, scikit-learn |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |
| Application | Streamlit |
| Versioning | Git / GitHub |

---

## ⚠️ Limitations

- Historical performance does not guarantee production performance
- Fraud patterns evolve — the model requires drift monitoring and periodic retraining
- Threshold choice is a business decision, not purely a statistical one
- False positives create customer friction; false negatives create direct financial loss
- A real deployment would need a feature store, real-time scoring infrastructure, and human-in-the-loop review

---




## 🔐 Data Privacy & Security

- Do not commit real financial or personally identifiable information to a public repository
- Keep raw transaction data out of version control (`data/` is `.gitignore`d)
- Review dataset licensing/redistribution terms before publishing
- Never commit API keys, credentials, or secrets

---

## 📜 License

This project is released under the **MIT License** — see `LICENSE` for details.

---

## 🙏 Acknowledgments

- Dataset: Machine Learning Group – ULB, via Kaggle
- XGBoost, SHAP, and Streamlit documentation and communities
