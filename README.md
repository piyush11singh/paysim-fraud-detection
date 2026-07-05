 Fraud Detection Classifier (PaySim)
 
Detecting fraudulent financial transactions in a large-scale, highly imbalanced mobile money transaction dataset, using transaction behavior and account balance patterns.
 
> **Status: Work in progress.** This README documents the current baseline model and the specific, identified next steps to improve it. It is presented transparently as part of an iterative modeling process rather than a finished result.
 
## Problem Statement
 
Fraudulent transactions make up a tiny fraction of all financial activity, but carry disproportionate cost and risk. This project builds a classifier to flag likely fraudulent transactions in near real time, while minimizing unnecessary false alarms that would burden a fraud review team.
 
## Dataset
 
- **Source:** PaySim synthetic financial transaction dataset (Kaggle)
- **Size:** 6,362,620 transactions, 11 columns, no missing values
- **Target variable:** `isFraud` (1 = fraudulent, 0 = legitimate)
- **Class balance:** Only 8,213 fraud cases out of 6.36M transactions (~0.13%) — extreme imbalance
**Features:** `step` (time), `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`
 
## Approach
 
### 1. Exploratory Data Analysis
 
- **Fraud is concentrated entirely in two transaction types:**
  | Type | Fraud rate |
  |---|---|
  | TRANSFER | 0.769% |
  | CASH_OUT | 0.184% |
  | CASH_IN / DEBIT / PAYMENT | 0% |
- Transaction amount distribution examined via log-scaled histograms and boxplots (filtered under 50K) split by fraud status
- Time-series view of fraud occurrences across `step` (time unit)
- Correlation analysis across balance and amount fields
### 2. Key Pattern Identified: The "Zero-Balance" Fraud Fingerprint
 
Filtering for transactions where the origin account's balance dropped to exactly zero after a TRANSFER or CASH_OUT surfaced **1,188,074 matching transactions** — a strong behavioral fingerprint consistent with known fraud patterns in this dataset (full balance drain in a single transaction).
 
### 3. Feature Engineering
 
- `balanceDiffOrig` = `oldbalanceOrg` − `newbalanceOrig`
- `balanceDiffDest` = `oldbalanceDest` − `newbalanceDest`
- `zero_after_transfer` — binary flag for the zero-balance pattern identified above *(next iteration)*
### 4. Baseline Model
 
- Logistic Regression with `class_weight="balanced"`, in a `scikit-learn` pipeline with `StandardScaler` (numeric) + `OneHotEncoder` (transaction type)
- **Note:** the baseline model was trained only on raw balance/amount columns — the engineered `balanceDiffOrig`/`balanceDiffDest` features were not yet included in this run (see Next Steps)
### 5. Baseline Results (held-out test set)
 
| Metric | Fraud class (1) |
|---|---|
| Precision | 0.02 |
| Recall | 0.94 |
| F1-score | 0.04 |
| Overall Accuracy | 94.7% |
| CV F1 (5-fold) | 0.043 |
 
**Confusion matrix:** 2,547 true positives / 163 false negatives / 111,605 false positives / 1,985,350 true negatives
 
### Honest Assessment of the Baseline
 
The baseline model catches the large majority of actual fraud (94% recall) but does so at the cost of an unworkable false-positive rate — flagging ~112K legitimate transactions for every ~2,547 real frauds caught. This result is documented here deliberately, rather than omitted, because it directly motivated the modeling changes below.
 
## Identified Fixes (Next Iteration)
 
1. **Include the already-engineered `balanceDiffOrig` / `balanceDiffDest` features** in the model — these were computed during EDA but excluded from the baseline's feature set
2. **Add the `zero_after_transfer` flag** as an explicit feature, based on the fraud fingerprint pattern discovered in EDA
3. **Replace Logistic Regression with a tree-based model** (Random Forest / XGBoost) to better capture non-linear fraud patterns and reduce the false-positive rate under extreme imbalance
4. **Evaluate using Precision-Recall AUC** rather than default-threshold F1, given the severity of the class imbalance
5. **Tune the classification threshold deliberately**, rather than using the default 0.5 cutoff, to find an operationally realistic precision/recall tradeoff
## Repository Structure
 
```
fraud-detection-classifier/
├── README.md
├── data/
│   └── Fraud_Dataset.csv
├── analysis.ipynb
└── requirements.txt
```
 
## Tech Stack
 
`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `joblib`
 
## Next Steps
 
- Retrain with full engineered feature set + tree-based model
- Precision-Recall curve analysis and threshold selection
- Business framing of the final precision/recall tradeoff (cost of false positive review vs. cost of missed fraud)
