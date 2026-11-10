# Customer Churn Prediction

Predicting whether a telecom customer will churn based on demographic, account, and service usage data.

## Project Overview

This project uses the **Telco Customer Churn** dataset to build a binary classification model that identifies customers at risk of leaving. Early detection of churn helps the business take retention actions.

**Target:** `Churn` (0 = No, 1 = Yes)

## Dataset

| Item | Details |
|------|---------|
| Source | Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn) |
| Original size | ~7,043 customers |
| Features | Demographics, services, contract type, charges |
| Target | `Churn` (Yes / No → encoded as 1 / 0) |
| Class balance | Imbalanced (~26% churn) |

### Main Features

- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account:** tenure, Contract, PaperlessBilling, PaymentMethod
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Charges:** MonthlyCharges, TotalCharges

### Feature Engineering

Additional features were created:

- `NumServices` – total number of subscribed services
- `SecurityServices` – count of security-related services
- `StreamingServices` – count of streaming services
- `HasInternet` – whether the customer has internet
- `HasSupportOrSecurity` – has tech support or online security
- `AvgMonthlySpend` – TotalCharges / (tenure + 1)
- `TenureGroup` – tenure bins
- `Contract_Internet`, `Contract_Payment` – interaction features
- `ChargePerService` – MonthlyCharges per service
- `NewCustomer_MonthlyContract` – new customer on month-to-month contract

## Project Structure

```
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── EDA.ipynb                 # Exploratory Data Analysis + Feature Engineering
├── churn_utils_data.py       # Data loading, encoding, scaling, resampling
├── churn_eval.py             # Threshold optimization
└── churn_train.py            # Random Forest training + evaluation
```

## Approach

1. **Exploratory Data Analysis**
   - Class distribution
   - Missing values and data cleaning (`TotalCharges`)
   - Categorical vs Churn analysis
   - Numerical feature distributions

2. **Preprocessing**
   - Train / Validation / Test split (stratified)
   - One-Hot Encoding for categorical features
   - SMOTE oversampling (`sampling_strategy=0.2`)

3. **Modeling**
   - Random Forest with `class_weight="balanced"`
   - Hyperparameter tuning with `GridSearchCV` (F1 scoring)
   - Decision threshold optimization on validation set

4. **Evaluation**
   - Classification Report
   - Balanced Accuracy
   - Confusion Matrix
   - Final metrics reported on both Validation and Test sets

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

# Run EDA and feature engineering
jupyter notebook EDA.ipynb

# Train model
python churn_train.py
```

## Notes

- Categorical features are encoded before applying SMOTE.
- Threshold is selected on the validation set to maximize F1-score.
- Final model and preprocessing objects are saved as `churn_rf_model.pkl`.
