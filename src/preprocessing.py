"""
Preprocessing pipeline for Student Burnout Classification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COL = "burnout_level"

NUMERIC_COLS = [
    "age",
    "daily_study_hours",
    "daily_sleep_hours",
    "screen_time_hours",
    "anxiety_score",
    "depression_score",
    "academic_pressure_score",
    "financial_stress_score",
    "social_support_score",
    "physical_activity_hours",
    "attendance_percentage",
    "cgpa",
]

CATEGORICAL_COLS = [
    "gender",
    "year",
    "course",
    "stress_level",
    "sleep_quality",
    "internet_quality",
]


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Удаляем ID-подобные колонки, если есть
    id_cols = [c for c in df.columns if "id" in c.lower()]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset info."""
    print("=== Shape ===")
    print(df.shape)
    print("\n=== dtypes ===")
    print(df.dtypes)
    print("\n=== Missing values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values")
    print("\n=== Duplicates ===")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\n=== Target distribution ===")
    print(df[TARGET_COL].value_counts(normalize=True).round(3))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {n_before - len(df)} duplicate rows")

    # 2. Convert expected numeric columns to float (coerce strings to NaN)
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    for col in numeric_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Fill missing values
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in numeric_present:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_present:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Remove outliers (IQR, 1%-99%)
    n_before = len(df)
    for col in numeric_present:
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        q1 = valid.quantile(0.01)
        q3 = valid.quantile(0.99)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]
    print(f"Removed {n_before - len(df)} outlier rows (IQR 1%-99%)")

    return df


def feature_engineering(df: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    """
    Add new features. If thresholds dict is provided, use it for binary flags
    (avoids data leakage when applied to val/test).
    """
    df = df.copy()

    # Academic pressure index: study_hours / (sleep_hours + 1) / cgpa
    if all(c in df.columns for c in ["daily_study_hours", "daily_sleep_hours", "cgpa"]):
        df["academic_pressure_index"] = (
            df["daily_study_hours"] / (df["daily_sleep_hours"] + 1)
        ) * (1 / (df["cgpa"] + 0.01))

    # Mental health composite
    mental_cols = [
        c
        for c in ["anxiety_score", "depression_score", "academic_pressure_score"]
        if c in df.columns
    ]
    if mental_cols:
        df["mental_health_score"] = df[mental_cols].mean(axis=1)

    # Lifestyle balance: physical activity vs screen time
    if "physical_activity_hours" in df.columns and "screen_time_hours" in df.columns:
        df["lifestyle_balance"] = (
            df["physical_activity_hours"] - df["screen_time_hours"]
        )

    # Low sleep flag
    if "daily_sleep_hours" in df.columns:
        df["is_sleep_deprived"] = (df["daily_sleep_hours"] < 6).astype(int)

    # High financial stress flag (example threshold)
    if "financial_stress_score" in df.columns:
        if thresholds is not None and "high_fin_stress" in thresholds:
            thr = thresholds["high_fin_stress"]
        else:
            thr = df["financial_stress_score"].quantile(0.75)
        df["is_high_financial_stress"] = (df["financial_stress_score"] >= thr).astype(
            int
        )
        if thresholds is None:
            thresholds = {}
        thresholds["high_fin_stress"] = thr

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}

    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_present:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL].astype(str))
    encoders[TARGET_COL] = le_target
    print(f"Target classes: {list(le_target.classes_)}")

    return df, encoders


def split_data(df: pd.DataFrame, val_size=0.15, test_size=0.15):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    train = X_train.copy()
    train[TARGET_COL] = y_train.values
    val = X_val.copy()
    val[TARGET_COL] = y_val.values
    test = X_test.copy()
    test[TARGET_COL] = y_test.values
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def scale_features(train, val, test, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = [c for c in NUMERIC_COLS if c in train.columns]
    scaler = StandardScaler()
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    val[numeric_cols] = scaler.transform(val[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])
    return train, val, test, scaler
