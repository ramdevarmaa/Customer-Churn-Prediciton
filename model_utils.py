import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb

FEATURES = [
    "Tenure", "CashbackAmount", "CityTier", "WarehouseToHome",
    "OrderAmountHikeFromlastYear", "DaySinceLastOrder", "SatisfactionScore",
    "NumberOfAddress", "NumberOfDeviceRegistered", "Complain",
    "OrderCount", "HourSpendOnApp", "MaritalStatus", "CouponUsed", "Gender"
]

FEATURE_LABELS = {
    "Tenure": "Tenure (months)",
    "CashbackAmount": "Cashback Amount ($)",
    "CityTier": "City Tier (1-3)",
    "WarehouseToHome": "Warehouse to Home (km)",
    "OrderAmountHikeFromlastYear": "Order Hike from Last Year (%)",
    "DaySinceLastOrder": "Days Since Last Order",
    "SatisfactionScore": "Satisfaction Score (1-5)",
    "NumberOfAddress": "Number of Addresses",
    "NumberOfDeviceRegistered": "Devices Registered",
    "Complain": "Has Complained (0/1)",
    "OrderCount": "Order Count",
    "HourSpendOnApp": "Hours Spent on App",
    "MaritalStatus": "Marital Status",
    "CouponUsed": "Coupons Used",
    "Gender": "Gender",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "churn_model.pkl")

# Mappings for categorical variables
MARITAL_MAPPING = {"Single": 0, "Married": 1, "Divorced": 2}
GENDER_MAPPING = {"Female": 0, "Male": 1}

def preprocess_input(df):
    """Preprocess input data to handle categorical variables."""
    df = df.copy()
    if "MaritalStatus" in df.columns:
        if df["MaritalStatus"].dtype == "object":
            df["MaritalStatus"] = df["MaritalStatus"].map(MARITAL_MAPPING)
            if df["MaritalStatus"].isna().any():
                unknown = df.loc[df["MaritalStatus"].isna(), "MaritalStatus"].unique()
                raise ValueError(f"Unknown MaritalStatus values: {list(unknown)}. Expected: {list(MARITAL_MAPPING.keys())}")
        df["MaritalStatus"] = df["MaritalStatus"].astype("category")
    if "Gender" in df.columns:
        if df["Gender"].dtype == "object":
            df["Gender"] = df["Gender"].map(GENDER_MAPPING)
            if df["Gender"].isna().any():
                unknown = df.loc[df["Gender"].isna(), "Gender"].unique()
                raise ValueError(f"Unknown Gender values: {list(unknown)}. Expected: {list(GENDER_MAPPING.keys())}")
        df["Gender"] = df["Gender"].astype("category")
    return df


def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)
    n_churn = int(n * 0.17)
    n_retained = n - n_churn

    def make_segment(size, churn):
        tenure = np.random.exponential(scale=20 if churn else 40, size=size).clip(1, 61)
        cashback = np.random.normal(150 if churn else 210, 40, size).clip(50, 400)
        city_tier = np.random.choice([1, 2, 3], size=size, p=[0.4, 0.2, 0.4])
        warehouse = np.random.normal(25 if churn else 15, 10, size).clip(5, 60)
        order_hike = np.random.normal(14 if churn else 20, 5, size).clip(5, 40)
        days_last = np.random.exponential(10 if churn else 4, size).clip(0, 46)
        satisfaction = np.random.choice([1, 2, 3, 4, 5], size=size,
                                        p=[0.3, 0.3, 0.2, 0.1, 0.1] if churn else [0.05, 0.1, 0.25, 0.35, 0.25])
        num_addr = np.random.randint(1, 10, size)
        num_devices = np.random.randint(1, 6, size)
        complain = np.random.binomial(1, 0.5 if churn else 0.1, size)
        order_count = np.random.poisson(2 if churn else 4, size).clip(1, 16)
        hours = np.random.choice([1, 2, 3, 4, 5], size=size,
                                  p=[0.2, 0.3, 0.3, 0.15, 0.05])
        marital = np.random.choice(["Single", "Married", "Divorced"], size=size)
        coupons = np.random.poisson(1 if churn else 2, size).clip(0, 10)
        gender = np.random.choice(["Female", "Male"], size=size)
        churn_col = np.ones(size, dtype=int) if churn else np.zeros(size, dtype=int)

        return pd.DataFrame({
            "Tenure": tenure, "CashbackAmount": cashback, "CityTier": city_tier,
            "WarehouseToHome": warehouse, "OrderAmountHikeFromlastYear": order_hike,
            "DaySinceLastOrder": days_last, "SatisfactionScore": satisfaction,
            "NumberOfAddress": num_addr, "NumberOfDeviceRegistered": num_devices,
            "Complain": complain, "OrderCount": order_count, "HourSpendOnApp": hours,
            "MaritalStatus": marital, "CouponUsed": coupons, "Gender": gender,
            "Churn": churn_col
        })

    df = pd.concat([make_segment(n_retained, False), make_segment(n_churn, True)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def train_model():
    df = generate_synthetic_data()
    X = df[FEATURES]
    y = df["Churn"]

    # Convert categorical columns to category dtype
    X["MaritalStatus"] = X["MaritalStatus"].astype("category")
    X["Gender"] = X["Gender"].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": roc_curve(y_test, y_prob),
        "feature_importance": dict(zip(FEATURES, model.feature_importances_)),
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics}, f)

    return model, metrics


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["metrics"]


def predict_single(model, input_dict):
    df = pd.DataFrame([input_dict])[FEATURES]
    df = preprocess_input(df)
    prob = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)
    return label, prob


def predict_batch(model, df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    X = df[FEATURES]
    X = preprocess_input(X)
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)
    result = df.copy()
    result["ChurnProbability"] = probs.round(3)
    result["ChurnPrediction"] = labels
    result["Risk"] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                             labels=["Low", "Medium", "High"])
    return result, None
