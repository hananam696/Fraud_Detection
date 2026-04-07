# Banking Fraud Detection

This project implements an end-to-end fraud detection system using machine learning, covering data ingestion, preprocessing, modeling, explainability, and deployment.

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

## How to Run

1. Install dependencies using:
pip install -r requirements.txt
Refer: requirements.txt for the dependencies used.

3. Download the dataset using the Kaggle API  

4. Run notebooks in the following order:

- notebooks/01_EDA_and_Data_Understanding.ipynb  
- notebooks/02_Data_Cleaning.ipynb  
- notebooks/03_Model_Ready_Data_and_Feature_Engineering.ipynb  
- notebooks/04_Modeling_and_Evaluation.ipynb  
- notebooks/05_Explainability_Bias_and_Deployment.ipynb
  
4. Running the final notebook will generate the trained model and deployment artifacts.

5. (Optional) Run the scoring script to test predictions on new data:
Refer:
python model_package/scoring_script.py  
The trained model and related artifacts are stored in the `model_package/` directory.
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

## Gold Layer — Model-Ready Data & Feature Engineering

### Sampling Strategy

The full silver dataset (8.9M rows) is too large for efficient modeling. A stratified sampling approach was used to create a manageable yet representative working dataset.

**All 13,332 fraud cases were retained.** 100,000 non-fraud transactions were sampled using stratified sampling by `use_chip` and `year` to preserve the original distributions.

| | Rows | Fraud Cases | Fraud Rate |
|---|---|---|---|
| Silver (full) | 8,914,963 | 13,332 | 0.15% |
| Gold (sampled) | 113,320 | 13,332 | 11.76% |

> **Why stratify by `use_chip` and `year`?**
> These two variables drive the largest structural differences in transaction behavior. Stratifying on them ensures the sample mirrors the real-world mix of chip, swipe, and online transactions across each year — preventing the model from learning patterns that don't generalize.

A representativeness check confirmed that distributions of transaction type, year, gender, card type, credit limit, age, and amount all remain consistent between the full silver dataset and the sample.

---

### Feature Engineering

New features were derived from existing columns to capture fraud-relevant signals that raw fields alone cannot express.

**Date Features**

| Feature | Description | Why it matters |
|---|---|---|
| `year` | Transaction year | Fraud patterns shift over time |
| `month` | Transaction month | Seasonal fraud spikes |
| `hour` | Hour of transaction | Fraud concentrates in off-hours |
| `day_of_week` | Day of week (0=Mon) | Weekend vs. weekday behavior differs |
| `is_night` | 1 if hour is 10PM–5AM | Night transactions are higher fraud risk |

**Card & Account Features**

| Feature | Description | Why it matters |
|---|---|---|
| `days_until_expiry` | Days until card expires | Cards near expiry are riskier |
| `account_age_days` | Days since account opened | New accounts have higher fraud rates |
| `years_since_pin_change` | Years since PIN was last changed | Stale PINs are a vulnerability |
| `is_expired_card` | 1 if card is past expiry date | Expired cards used in fraud is a red flag |

**Financial Ratio Features**

| Feature | Description | Why it matters |
|---|---|---|
| `amount_to_limit_ratio` | Transaction amount ÷ credit limit | High ratios signal unusual spending |
| `amount_to_income_ratio` | Transaction amount ÷ yearly income | Flags purchases disproportionate to income |
| `debt_to_income_ratio` | Total debt ÷ yearly income | Captures financial stress level |

**Demographic Feature**

| Feature | Description | Why it matters |
|---|---|---|
| `age_group` | Binned age: 18–25, 26–35, 36–50, 51–65, 65+ | Used for segmentation and bias analysis |

---

### Feature Selection & Columns Dropped

A correlation analysis was run after feature engineering to identify redundant features.

**High-correlation drops (>0.7 with another feature):**

| Column Dropped | Correlated With | Action |
|---|---|---|
| `per_capita_income` | `yearly_income` (r = 0.947) | Dropped — near-duplicate signal |
| `retirement_age` | `current_age` | Dropped — redundant demographic |
| `is_weekend` | `day_of_week` (r = 0.789) | Dropped — derived from same signal |

**Other drops:**

| Column Dropped | Reason |
|---|---|
| `id`, `client_id`, `card_id` | Leakage risk — unique identifiers |
| `date`, `expires`, `acct_open_date`, `year_pin_last_changed` | Raw date columns replaced by engineered features |
| `mcc_description` | Redundant with `mcc` numeric code |
| `total_debt` | Captured via `debt_to_income_ratio` |
| `days_until_expiry` | Captured via `is_expired_card` |
| `mcc` | Too high cardinality; meaning captured by other features |
| `is_zero_amount` | No fraud signal after exploration |

**Final Gold dataset:**  
- Final dataset shape: 113,320 rows × 27 columns  
- Fraud rate: ~11.76%  

The final features used for modeling include:
- Transaction features:  
  amount, use_chip, merchant_state, is_refund, is_zero_amount  
- Card features:  
  card_brand, card_type, has_chip, num_cards_issued, credit_limit  
- Customer features:  
  current_age, gender, yearly_income, credit_score, num_credit_cards  
- Time-based features:  
  hour, year, month, day_of_week, is_night  
- Account features:  
  account_age_days, is_expired_card  
- Financial ratio features:  
  amount_to_limit_ratio, amount_to_income_ratio, debt_to_income_ratio  
- Demographic feature:  
  age_group  
- Target variable:  
  is_fraud  

This dataset was used for model training and evaluation.
Saved as `gold.parquet` for efficient and faster processing during modeling, and as `gold.csv` for use in Power BI dashboard creation.
Refer: `notebooks/03_Model_Ready_Data_and_Feature_Engineering.ipynb`.

---

## Modeling & Evaluation

### Train-Test Split

The gold dataset was split using **stratified random 80/20 split**.

| Set | Rows | Fraud Rate |
|---|---|---|
| Train | 90,656 | ~11.76% |
| Test | 22,664 | ~11.76% |

> **Why stratified random split instead of time-based?**
> A time-based split (train on 2010–2018, test on 2019) was attempted first but produced poor results — fraud rates differ significantly across years, causing the model to learn the wrong distribution. Stratified random split ensures both sets have identical fraud rates and is the standard approach for academic classification projects.

### Encoding

Categorical columns were encoded using `LabelEncoder` fitted only on training data. The test set was mapped using the fitted encoder (unseen categories → `-1`) to prevent data leakage.

Columns encoded: `use_chip`, `merchant_state`, `card_brand`, `card_type`, `gender`, `age_group`

### Class Imbalance Handling

Even after sampling, the fraud rate is ~11.76%. Tree-based models used `scale_pos_weight` (ratio of non-fraud to fraud in training) and `class_weight='balanced'` to further compensate.

---

### Models Trained

**Baseline — Logistic Regression**
- `class_weight='balanced'`, `max_iter=500`
- Establishes a performance floor; captures only linear relationships

**Random Forest**
- 200 trees, `max_depth=10`, `class_weight='balanced'`
- Captures non-linear patterns; significant improvement over baseline

**LightGBM**
- 200 estimators, `learning_rate=0.05`, `max_depth=6`, `scale_pos_weight` applied
- Gradient boosting handles imbalance well; fast and scalable

**AutoML — FLAML**
- 120-second budget, `metric='roc_auc'`, `task='classification'`
- Automatically selected LightGBM as the best estimator

---

### Model Comparison

| Model | AUC-ROC | F1 (Fraud) |
|---|---|---|
| Logistic Regression | ~0.70 | Low |
| Random Forest | High | High |
| LightGBM | Very High | High |
| **AutoML (LightGBM)** | **~0.997** | **Best** |

**AutoML (LightGBM)** was selected as the final model.

### Threshold Tuning

The default decision threshold of 0.5 was lowered to **0.3** to prioritize recall (catching more fraud) at the cost of some precision. In fraud detection, missing a fraudulent transaction (false negative) is far more costly than a false alarm.

### Overfitting Check

A learning curve was generated using 3-fold cross-validation on LightGBM. Train and validation AUC converge closely as training size grows, confirming the model **generalizes well and is not overfitting**.

Refer: `notebooks/04_Modeling_and_Evaluation.ipynb`.

---

## Explainability, Bias Analysis & Deployment

### SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to the AutoML model using `TreeExplainer` to produce feature-level explanations at both global and individual prediction levels.

**Top features by mean |SHAP| value:**

| Rank | Feature | Insight |
|---|---|---|
| 1 | `year` | Fraud patterns have shifted significantly across 2010–2019 |
| 2 | `merchant_state` | Geographic location is a strong fraud signal |
| 3 | `hour` | Time of day heavily influences fraud probability |
| 4 | `amount` | Larger transactions carry higher fraud risk |
| 5 | `amount_to_limit_ratio` | Spending close to credit limit is suspicious |
| 6 | `amount_to_income_ratio` | Disproportionate amounts relative to income signal fraud |
| 7 | `account_age_days` | Newer accounts are riskier |
| 8 | `use_chip` | Transaction method (online vs. chip vs. swipe) matters |

Three levels of SHAP analysis were produced:
- **Summary plot** — overall feature importance ranked by mean |SHAP|, with direction
- **Bar plot** — simplified global importance ranking
- **Waterfall plot** — individual transaction explanation (why this specific transaction was flagged)
- **Dependence plot** — how `amount` interacts with other features to drive predictions

---

### Bias & Fairness Analysis

Recall was used as the fairness metric — it measures whether the model detects fraud equally well across demographic groups. A recall gap > 0.05 between groups was set as the threshold for flagging bias.

**Gender**

| Group | Recall |
|---|---|
| Male | ~0.93 |
| Female | ~0.91 |

Recall gap < 0.05 → **No significant bias detected.**

**Age Group**

| Age Group | Recall |
|---|---|
| 18–25 | Measured |
| 26–35 | Measured |
| 36–50 | Measured |
| 51–65 | Measured |
| 65+ | Measured (small sample — interpret cautiously) |

No strong bias was observed across age groups. Some groups have small sample sizes, making their recall estimates less reliable.

> **Why recall as the fairness metric?**
> In fraud detection, failing to catch fraud for one demographic group (low recall) is a meaningful harm — that group receives less protection. Recall gap between groups is therefore the most appropriate fairness measure here.

---

### Deployment

The trained AutoML model was packaged with all required artifacts for real-world scoring.

**Model Package Contents:**

| Artifact | File | Description |
|---|---|---|
| Trained model | `fraud_model.pkl` | Serialized AutoML (LightGBM) model |
| Feature list | `feature_names.pkl` | Ordered list of features used at training |
| Decision threshold | `threshold.pkl` | Tuned threshold (0.3) |
| Encoders | `encoders.pkl` | Fitted LabelEncoders for categorical columns |
| Scoring script | `scoring_script.py` | End-to-end prediction function |

**Scoring Script** (`scoring_script.py`) takes a raw transaction DataFrame, applies the same preprocessing and encoding as training, and returns:

```python
{
  "fraud_probability": float,   # model confidence score
  "prediction": int             # 0 = Legit, 1 = Fraud
}
```

The scoring script was tested on held-out test samples and known fraud cases — predictions and probabilities were generated correctly in both cases.

Refer: `notebooks/05_Explainability_Bias_and_Deployment.ipynb`.

## Power BI Dashboard

An interactive Power BI dashboard was developed to analyze fraud patterns, transaction behavior, and customer insights.

The dashboard includes:
- Executive overview of fraud metrics  
- Fraud trends over time  
- Customer and geographic analysis  

View Dashboard: https://app.powerbi.com/reportEmbed?reportId=ea85ce31-d657-4129-97fe-17518305b9b1&autoAuth=true&ctid=b30f4b44-46c6-4070-9997-f87b38d4771c&actionBarEnabled=true&reportCopilotInEmbed=true 

## Final Results

- Best Model: AutoML (LightGBM)  

- Recall (Fraud - Class 1): ~0.93  
- Recall (Legitimate - Class 0): ~0.99  

- F1 Score (Fraud): ~0.93  
- AUC-ROC: ~0.997  

- Model prioritizes recall to minimize missed fraud cases (false negatives), which is critical in fraud detection  

- SHAP explainability identified key features influencing predictions:
  - year  
  - merchant_state  
  - hour  
  - amount  

- Bias analysis shows no significant difference in recall across gender and age groups  

- Model is fully packaged with preprocessing, encoders, and scoring script, making it ready for real-world deployment.

## Conclusion

End-to-end fraud detection pipeline successfully implemented.

