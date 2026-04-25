"""
Basic pipeline tests.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    clean_data,
    encode_features,
    feature_engineering,
    split_data,
)
from src.modeling import evaluate_model, get_X_y

RANDOM_STATE = 42


def make_dummy_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    return pd.DataFrame(
        {
            # --- числовые признаки (NUMERIC_COLS) ---
            "age": rng.integers(18, 30, n),
            "daily_study_hours": rng.uniform(1, 10, n).round(1),
            "daily_sleep_hours": rng.uniform(4, 10, n).round(1),
            "screen_time_hours": rng.uniform(1, 12, n).round(1),
            "anxiety_score": rng.integers(0, 10, n),
            "depression_score": rng.integers(0, 10, n),
            "academic_pressure_score": rng.integers(0, 10, n),
            "financial_stress_score": rng.integers(0, 10, n),
            "social_support_score": rng.integers(0, 10, n),
            "physical_activity_hours": rng.uniform(0, 5, n).round(1),
            "attendance_percentage": rng.uniform(50, 100, n).round(1),
            "cgpa": rng.uniform(2.0, 4.0, n).round(2),
            # --- категориальные признаки (CATEGORICAL_COLS) ---
            "gender": rng.choice(["Male", "Female"], n),
            "year": rng.choice(["1st", "2nd", "3rd", "4th"], n),
            "course": rng.choice(["CS", "Math", "Physics"], n),
            "stress_level": rng.choice(["Low", "Medium", "High"], n),
            "sleep_quality": rng.choice(["Good", "Average", "Poor"], n),
            "internet_quality": rng.choice(["Good", "Poor"], n),
            # --- целевая переменная ---
            "burnout_level": rng.choice(["Low", "Medium", "High"], n),
        }
    )


class TestCleaning:
    def test_no_duplicates_after_clean(self):
        df = make_dummy_df()
        df = pd.concat([df, df.iloc[:10]])  # add duplicates
        cleaned = clean_data(df)
        assert cleaned.duplicated().sum() == 0

    def test_no_missing_after_clean(self):
        df = make_dummy_df()
        df.loc[0:5, "cgpa"] = np.nan
        cleaned = clean_data(df)
        assert cleaned["cgpa"].isnull().sum() == 0

    def test_shape_after_clean(self):
        df = make_dummy_df(200)
        cleaned = clean_data(df)
        assert len(cleaned) > 0
        assert cleaned.shape[1] == df.shape[1]


class TestFeatureEngineering:
    def test_new_columns_created(self):
        df = make_dummy_df()
        df_fe = feature_engineering(df)
        assert "mental_health_score" in df_fe.columns
        assert "lifestyle_balance" in df_fe.columns
        assert "is_sleep_deprived" in df_fe.columns

    def test_no_nans_in_new_features(self):
        df = make_dummy_df()
        df_fe = feature_engineering(df)
        new_cols = ["mental_health_score", "lifestyle_balance", "is_sleep_deprived"]
        for col in new_cols:
            if col in df_fe.columns:
                assert df_fe[col].isnull().sum() == 0, f"{col} has NaNs"


class TestSplit:
    def test_split_sizes(self):
        df = make_dummy_df(500)
        df, _ = encode_features(df)
        train, val, test = split_data(df)
        total = len(train) + len(val) + len(test)
        assert abs(total - len(df)) <= 1  # rounding tolerance

    def test_no_leakage_between_splits(self):
        df = make_dummy_df(500)
        df, _ = encode_features(df)
        train, val, test = split_data(df)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0


class TestModeling:
    def test_evaluate_returns_metrics(self):
        df = make_dummy_df(300)
        df_enc, _ = encode_features(df)
        train, val, _ = split_data(df_enc)
        X_train, y_train = get_X_y(train)
        X_val, y_val = get_X_y(val)
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val, "val")
        assert "weighted_f1" in metrics
        assert 0.0 <= metrics["weighted_f1"] <= 1.0
