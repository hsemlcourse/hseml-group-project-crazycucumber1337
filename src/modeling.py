"""
Model training and evaluation for Student Burnout Classification.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.preprocessing import TARGET_COL

RANDOM_STATE = 42


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "val",
) -> dict:
    """
    Evaluate a trained model and return metrics dict.
    """
    y_pred = model.predict(X)

    metrics = {
        "split": split_name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "weighted_f1": round(f1_score(y, y_pred, average="weighted"), 4),
        "macro_f1": round(f1_score(y, y_pred, average="macro"), 4),
    }

    # ROC-AUC (needs predict_proba)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        metrics["roc_auc_ovr"] = round(
            roc_auc_score(y, y_proba, multi_class="ovr", average="weighted"), 4
        )

    print(f"\n=== {split_name.upper()} ===")
    for k, v in metrics.items():
        if k != "split":
            print(f"  {k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    return metrics


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """Run k-fold cross-validation and return mean/std of weighted F1."""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    result = {
        "cv_f1_mean": round(scores.mean(), 4),
        "cv_f1_std": round(scores.std(), 4),
    }
    print(f"CV F1 (weighted): {result['cv_f1_mean']} ± {result['cv_f1_std']}")
    return result


def save_model(model, path: str) -> None:
    """Save model with joblib."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    """Load model from joblib file."""
    return joblib.load(path)


def build_experiment_table(results: list[dict]) -> pd.DataFrame:
    """
    Build a summary table from a list of experiment result dicts.
    Each dict should have keys: model_name, hypothesis, weighted_f1, accuracy, notes
    """
    return pd.DataFrame(results).sort_values("weighted_f1", ascending=False)
