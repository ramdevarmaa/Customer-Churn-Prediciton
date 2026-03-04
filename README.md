# ChurnSight — Customer Churn Prediction App

A production-grade Streamlit application for predicting customer churn using XGBoost.

## Features

- **Overview Dashboard** — Model KPIs, ROC curve, churn distribution, feature importances
- **Single Prediction** — Interactive form to predict churn for one customer with gauge visualization
- **Model Insights** — Confusion matrix, metric bar chart, probability distributions, violin plots
- **Batch Analysis** — Upload a CSV to predict churn for thousands of customers at once, with downloadable results

## Setup

```bash
# 1. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will automatically train an XGBoost model on synthetic e-commerce data on first run and cache it for subsequent sessions.

## Project Structure

```
churn_app/
├── app.py              # Main Streamlit entry point + shared CSS
├── model_utils.py      # Model training, prediction utilities
├── requirements.txt
├── models/             # Auto-generated model pickle
└── pages/
    ├── overview.py     # Dashboard page
    ├── predict.py      # Single prediction page
    ├── insights.py     # Model insights page
    └── batch.py        # Batch CSV analysis page
```

## CSV Format for Batch Analysis

Your CSV must contain these columns:
`Tenure, CashbackAmount, CityTier, WarehouseToHome, OrderAmountHikeFromlastYear, DaySinceLastOrder, SatisfactionScore, NumberOfAddress, NumberOfDeviceRegistered, Complain, OrderCount, HourSpendOnApp, MaritalStatus, CouponUsed, Gender`

Or just click **"Generate & Analyze Sample Data"** in the Batch Analysis page to test with synthetic data.

## Model Details

- **Algorithm**: XGBoost Classifier
- **Imbalance Handling**: `scale_pos_weight` parameter
- **Key Features**: Tenure, CashbackAmount, DaySinceLastOrder, SatisfactionScore, Complain
- **Typical Performance**: ~92% Accuracy, ~0.85+ ROC AUC
