# Banking Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent banking transactions,
covering data ingestion, cleaning, feature engineering, modeling, explainability, and fairness analysis.

| | |
|---|---|
| **Domain** | Finance / Banking |
| **Dataset** | CaixaBank Financial Transactions (2010–2019) |
| **Problem Type** | Binary Classification |
| **Target** | `is_fraud` (0 = Legitimate, 1 = Fraudulent) |
| **Source** | [https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)|

---
## Project Pipeline

Data Collection (Kaggle) → Data Understanding & Initial EDA → Data Cleaning & Merging + EDA with target (feature–target relationships) → Feature Engineering → Train-Test Split → Encoding → Model Training → Model Evaluation → SHAP Explainability → Bias & Fairness Analysis → Deployment (Model Packaging + Scoring Script) → Power BI Dashboard
---

## Bronze Layer — Raw Data Ingestion and EDA

### Data Source
Dataset downloaded from Kaggle via the Kaggle API.

### Files Loaded

| File | Raw Shape | Format | Description |
|------|-----------|--------|-------------|
| `transactions_data.csv` | (13,305,915 × 12) | CSV | All banking transactions 2010–2019 |
| `cards_data.csv` | (6,146 × 13) | CSV | Card details per customer |
| `users_data.csv` | (2,000 × 14) | CSV | Customer demographics |
| `train_fraud_labels.json` | (8,914,963 × 2) | JSON | Fraud labels — recovered from corrupted JSON via regex |
| `mcc_codes.json` | (109 × 2) | JSON | Merchant Category Code descriptions |

All 5 files were loaded without modification and saved as `.parquet` for faster downstream access.

### Key Observation — Label Coverage
Only **67% of transactions (8.9M of 13.3M)** have fraud labels. The remaining 4.4M transactions are unlabeled and cannot be used for supervised modeling.

---

## EDA Findings

### Class Imbalance
- Only **0.15% of transactions are fraud** — 1 in every 667
- Standard accuracy is meaningless: a model predicting "no fraud" always would score 99.85%
- Metrics used: **AUC-ROC + F1 Score** instead of accuracy

### Transactions
- `amount` is stored as a string with `$` prefix — requires type conversion
- **4.96% of transactions have negative amounts** — these are refunds, not fraud
- `use_chip` breakdown: Swipe (52%) · Chip (36%) · Online (12%)
- `merchant_city` has 12,492 unique values — too high cardinality for modeling
- `merchant_state` has 1,563,700 missing values — confirmed to be 100% online transactions

### Cards
- `card_number` and `cvv` present — must be dropped (PII)
- `credit_limit` stored as string with `$` prefix
- `card_on_dark_web` has only 1 unique value — zero predictive signal
- `has_chip`: 5,500 chip-enabled vs 646 non-chip cards

### Users
- Age ranges from 18 to 101 · Credit score from 480 to 850
- Gender: Female 50.8% · Male 49.2% — balanced
- Income columns stored as strings with `$` — require cleaning
- `birth_year`, `birth_month`, `address`, `latitude`, `longitude` — redundant, dropped

### Fraud Labels
- 13,332 fraud cases out of 8,914,963 labeled transactions
- Imbalance ratio: **1 fraud per 667 legitimate transactions**
- No duplicate IDs, no missing values

Refer: `notebooks/01_EDA_and_Data_Understanding.ipynb`.

---

## Silver Layer — Data Cleaning & Merging

Each file was cleaned independently before merging.

### Transactions Cleaning

| Column | Issue | Action |
|--------|-------|--------|
| `amount` | String with `$`, negative values for refunds | Removed `$`, created `is_refund` flag, took absolute value |
| `is_zero_amount` | 10,639 zero-amount transactions | Created flag column |
| `merchant_state` | 1,563,700 missing values | Filled with `"ONLINE"` — confirmed all missing states are online transactions |
| `errors` | 98% NaN, 22 compound categories | Filled NaN → `"No Error"`, simplified to first error only (8 categories) |
| `zip` | 24,586 unique values, zero fraud signal | Dropped — high cardinality, memorization risk |
| `merchant_id` | 74,831 unique values | Dropped — too high cardinality |
| `merchant_city` | 12,492 unique values | Dropped — not useful for modeling |

> Note: `errors` was **not** dropped due to leakage — `Bad CVV` is a point-of-sale signal,
> not a derived fraud label. It was ultimately dropped later in the Gold layer during feature selection.

**Final shape:** (13,305,915 × 10) · No missing values

### Cards Cleaning

| Column | Action |
|--------|--------|
| `card_number`, `cvv` | Dropped — sensitive PII |
| `card_on_dark_web` | Dropped — only 1 unique value, no signal |
| `credit_limit` | Removed `$`, converted to float |
| `has_chip` | Binary encoded: YES → 1, NO → 0 |
| `acct_open_date`, `expires` | Parsed to datetime |

**Final shape:** (6,146 × 10) · No missing values

### Users Cleaning

| Column | Action |
|--------|--------|
| `birth_year`, `birth_month` | Dropped — redundant with `current_age` |
| `address`, `latitude`, `longitude` | Dropped — not useful for modeling |
| `per_capita_income`, `yearly_income`, `total_debt` | Removed `$`, converted to float |

**Final shape:** (2,000 × 9) · No missing values

### MCC Codes
No cleaning required — used as a reference lookup table.

All cleaned files were saved as `.parquet` for reusability.

---

### Merging

All files were merged using shared IDs in a sequential left-join strategy:
```python
df_silver = transactions_clean.merge(labels_bronze, on='id', how='inner')  # inner join on fraud labels
df_silver = df_silver.merge(cards_clean,  left_on='card_id',   right_on='id', how='left')
df_silver = df_silver.merge(users_clean,  left_on='client_id', right_on='id', how='left')
df_silver = df_silver.merge(mcc_clean,    on='mcc',                           how='left')
```

| Step | Shape | Fraud Rate |
|------|-------|------------|
| After joining labels | (8,914,963 × 11) | 0.1495% |
| After joining cards | (8,914,963 × 19) | 0.1495% |
| After joining users | (8,914,963 × 27) | 0.1495% |
| After joining MCC | (8,914,963 × 28) | 0.1495% |

> **Why does 13.3M rows shrink to 8.9M?**
> The first join is an **inner join** with the fraud labels file, which only covers 67% of transactions.
> The 4.4M unlabeled transactions are excluded here because without a fraud label, they cannot
> be used for supervised learning. The subsequent joins (cards, users, MCC) are left joins on IDs
> that exist for every transaction, so the row count stays stable at 8.9M.

**Final Silver dataset:** 8,914,963 rows × 28 columns · 13,332 fraud cases · No missing values

Refer: `notebooks/02_Data_Cleaning.ipynb`.



