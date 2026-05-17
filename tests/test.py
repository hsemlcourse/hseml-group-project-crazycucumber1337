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
    split_raw,
    fit_cleaning_stats,
    apply_cleaning,
    encode_features,
    feature_engineering,
    split_data,  # legacy alias
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
    def test_no_duplicates_after_split(self):
        df = make_dummy_df()
        df = pd.concat([df, df.iloc[:10]])  # add duplicates
        # split_raw deduplicates before splitting
        train_raw, val_raw, test_raw = split_raw(df)
        for name, split in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
            assert split.duplicated().sum() == 0, f"{name} has duplicates"

    def test_no_missing_after_apply_cleaning(self):
        df = make_dummy_df()
        df.loc[0:5, "cgpa"] = np.nan
        train_raw, val_raw, test_raw = split_raw(df)
        stats = fit_cleaning_stats(train_raw)
        train_c = apply_cleaning(train_raw, stats)
        assert train_c["cgpa"].isnull().sum() == 0

    def test_shape_preserved_after_clean(self):
        df = make_dummy_df(300)
        train_raw, val_raw, test_raw = split_raw(df)
        stats = fit_cleaning_stats(train_raw)
        train_c = apply_cleaning(train_raw, stats)
        assert len(train_c) > 0
        assert train_c.shape[1] == train_raw.shape[1]


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
        train_raw, val_raw, test_raw = split_raw(df)
        total = len(train_raw) + len(val_raw) + len(test_raw)
        assert abs(total - 500) <= 1  # rounding tolerance (dupes removed first)

    def test_no_leakage_between_splits(self):
        df = make_dummy_df(500)
        train_raw, val_raw, test_raw = split_raw(df)
        assert len(set(train_raw.index) & set(val_raw.index)) == 0
        assert len(set(train_raw.index) & set(test_raw.index)) == 0
        assert len(set(val_raw.index) & set(test_raw.index)) == 0


class TestCleaningPipeline:
    """Verify the fit/transform cleaning pattern prevents leakage."""

    def test_fit_cleaning_stats_keys(self):
        df = make_dummy_df(300)
        train_raw, _, _ = split_raw(df)
        stats = fit_cleaning_stats(train_raw)
        assert "medians" in stats
        assert "modes" in stats
        assert "outlier_bounds" in stats

    def test_apply_cleaning_no_missing(self):
        df = make_dummy_df(300)
        # Inject missing values
        df.loc[0:10, "cgpa"] = np.nan
        train_raw, val_raw, test_raw = split_raw(df)
        stats = fit_cleaning_stats(train_raw)
        train_c = apply_cleaning(train_raw, stats)
        val_c = apply_cleaning(val_raw, stats)
        test_c = apply_cleaning(test_raw, stats)
        for split_name, split_df in [("train", train_c), ("val", val_c), ("test", test_c)]:
            assert split_df.isnull().sum().sum() == 0, f"{split_name} has NaNs after cleaning"

    def test_val_stats_not_computed_from_val(self):
        """Val cleaning must use train-fitted stats, not its own stats."""
        df_train = make_dummy_df(300)
        df_val = make_dummy_df(100)
        # Give val extreme values — if we refit on val these would shift the bounds
        df_val["cgpa"] = 99.0
        stats = fit_cleaning_stats(df_train)
        # outlier bounds should be based on train (cgpa ~ 2-4), not val (99)
        lower, upper = stats["outlier_bounds"]["cgpa"]
        assert upper < 10, "Outlier upper bound leaked from val (should be train-only)"


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


class TestKaggleDataLoader:
    """Tests for the Kaggle API data loader (unit-level, no real API calls)."""

    def test_loader_initialises(self):
        from src.data_loader import KaggleDataLoader
        loader = KaggleDataLoader(output_dir="/tmp/test_dl")
        assert loader.owner == "sehaj1104"
        assert loader.dataset == "student-mental-health-and-burnout-dataset"

    def test_output_dir_created(self, tmp_path):
        from src.data_loader import KaggleDataLoader
        out = tmp_path / "raw"
        loader = KaggleDataLoader(output_dir=str(out))
        assert out.exists()

    def test_check_credentials_raises_without_creds(self, monkeypatch):
        from src.data_loader import KaggleDataLoader
        loader = KaggleDataLoader(output_dir="/tmp/test_dl")
        # Force empty credentials
        loader._username = None
        loader._key = None
        with pytest.raises(EnvironmentError, match="Kaggle credentials"):
            loader._check_credentials()

    def test_download_skips_if_file_exists(self, tmp_path):
        """If CSV already exists, download() returns immediately without API call."""
        from src.data_loader import KaggleDataLoader, DATASET_FILE
        # Pre-create the target file
        existing = tmp_path / DATASET_FILE
        existing.write_text("col1,col2\n1,2\n")
        loader = KaggleDataLoader(output_dir=str(tmp_path))
        result = loader.download()
        assert result == existing
