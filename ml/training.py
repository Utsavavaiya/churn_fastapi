import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import os
from datetime import datetime

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "random_forest":
        model = RandomForestClassifier()
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed. Please install it to use this model.")
        model = XGBClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary" if len(set(y))==2 else "weighted"),
        "recall": recall_score(y_test, y_pred, average="binary" if len(set(y))==2 else "weighted"),
        "f1_score": f1_score(y_test, y_pred, average="binary" if len(set(y))==2 else "weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()  # Use .tolist() not .toList()
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    return metrics, model_filename

def train_models(df, target_col, model_names):
    X = df.drop(columns=[target_col, "CustomerID"])
    y = df[target_col]
    results = {}
    model_files = {}
    for model_name in model_names:
        metrics, model_filename = train_and_evaluate(X, y, model_name)
        results[model_name] = metrics
        model_files[model_name] = model_filename
    return results, model_files
