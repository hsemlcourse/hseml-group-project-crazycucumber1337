"""
Preprocessing pipeline for Student Burnout Classification.

Data leakage prevention strategy
---------------------------------
ALL statistics (medians, modes, outlier bounds, quantile thresholds,
scaler parameters, one-hot column set) are computed **exclusively on the
training split** and then applied to val/test.  The correct execution order
enforced in the notebooks is:

  1. load_data()          – load raw CSV, normalise column names
  2. split_raw()          – stratified 70/15/15 split on the **raw** DataFrame
                            (before any cleaning), using only the target column
  3. fit_cleaning_stats() – compute medians/modes/IQR bounds from train_raw
  4. apply_cleaning()     – apply those pre-computed stats to every split
  5. feature_engineering()– create new features (thresholds fitted on train)
  6. encode_features()    – one-hot + LabelEncoder fitted on train
  7. scale_features()     – StandardScaler fitted on train, applied everywhere
"""

import joblib
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


# ── Loading ────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset and normalise column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
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


# ── Step 1: split RAW data (before any cleaning) ───────────────────────────────

def split_raw(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """
    Stratified split of the **raw** DataFrame (no cleaning applied yet).

    Splitting before cleaning is the only way to guarantee that imputation
    medians, mode fills and IQR outlier bounds are never contaminated by
    val/test observations.

    Returns
    -------
    train_raw, val_raw, test_raw – three DataFrames, all still unprocessed.
    """
    # Deduplicate first so we don't split duplicates across train and val/test.
    n_before = len(df)
    df = df.drop_duplicates()
    if n_before - len(df):
        print(f"Removed {n_before - len(df)} duplicate rows before split")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    def _reassemble(X, y):
        out = X.copy()
        out[TARGET_COL] = y.values
        return out

    train_raw = _reassemble(X_train, y_train)
    val_raw = _reassemble(X_val, y_val)
    test_raw = _reassemble(X_test, y_test)

    print(f"Raw split — Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")
    return train_raw, val_raw, test_raw


# ── Step 2: fit cleaning statistics on train, apply to all splits ──────────────

def fit_cleaning_stats(train_raw: pd.DataFrame) -> dict:
    """
    Compute all cleaning statistics from the **training split only**.

    Returns a ``cleaning_stats`` dict that can be passed to
    ``apply_cleaning()`` for any split (train / val / test).

    Statistics computed:
    - ``medians``       : median of every numeric column (for NaN imputation)
    - ``modes``         : mode of every categorical column (for NaN imputation)
    - ``outlier_bounds``: (lower, upper) IQR-based bounds per numeric column
    """
    df = train_raw.copy()

    # Cast numeric columns before computing stats
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    for col in numeric_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]

    medians = {col: df[col].median() for col in numeric_present}
    modes = {col: df[col].mode()[0] for col in cat_present if not df[col].mode().empty}

    outlier_bounds = {}
    for col in numeric_present:
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        q1, q3 = valid.quantile(0.01), valid.quantile(0.99)
        iqr = q3 - q1
        outlier_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    stats = {
        "medians": medians,
        "modes": modes,
        "outlier_bounds": outlier_bounds,
        "numeric_present": numeric_present,
        "cat_present": cat_present,
    }
    print(
        f"Cleaning stats fitted on train ({len(train_raw)} rows) — "
        f"{len(numeric_present)} numeric, {len(cat_present)} categorical columns"
    )
    return stats


def apply_cleaning(df: pd.DataFrame, cleaning_stats: dict) -> pd.DataFrame:
    """
    Apply pre-computed cleaning statistics to a DataFrame.

    No statistics are recomputed here — only the values from
    ``cleaning_stats`` (which were fitted on train) are used.
    This makes the function safe to call on val and test without leakage.
    """
    df = df.copy()

    numeric_present = cleaning_stats["numeric_present"]
    cat_present = cleaning_stats["cat_present"]
    medians = cleaning_stats["medians"]
    modes = cleaning_stats["modes"]
    outlier_bounds = cleaning_stats["outlier_bounds"]

    # 1. Cast numeric columns
    for col in numeric_present:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Impute missing values with train-fitted medians / modes
    for col in numeric_present:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(medians[col])
    for col in cat_present:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(modes.get(col, df[col].mode()[0]))

    # 3. Clip / remove outliers using train-fitted IQR bounds
    n_before = len(df)
    for col, (lower, upper) in outlier_bounds.items():
        if col in df.columns:
            df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]
    removed = n_before - len(df)
    if removed:
        print(f"  Removed {removed} outlier rows (train-fitted IQR bounds)")

    return df


# ── Legacy helper kept for backward compatibility ──────────────────────────────
# Callers that pass a single DataFrame (e.g. quick experiments) still work,
# but the docstring now makes the leakage risk explicit.

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    .. deprecated::
        Prefer ``split_raw()`` → ``fit_cleaning_stats()`` → ``apply_cleaning()``
        to avoid data leakage.  This helper computes statistics on whatever
        DataFrame is passed in, which causes leakage if that DataFrame contains
        val/test rows.

    Kept for backward compatibility with quick one-off experiments only.
    """
    stats = fit_cleaning_stats(df)
    return apply_cleaning(df, stats)


# ── Feature engineering ────────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    """
    Add derived features.

    ``thresholds`` must be a dict computed on the **training split** and
    re-used for val/test — avoids leakage for quantile-based binary flags.

    Pass an empty dict ``{}`` on the first call (train); the function
    populates it in-place.  Pass the same dict on subsequent calls (val/test).
    """
    df = df.copy()

    if all(c in df.columns for c in ["daily_study_hours", "daily_sleep_hours", "cgpa"]):
        df["academic_pressure_index"] = (
            df["daily_study_hours"] / (df["daily_sleep_hours"] + 1)
        ) * (1 / (df["cgpa"] + 0.01))

    mental_cols = [
        c for c in ["anxiety_score", "depression_score", "academic_pressure_score"]
        if c in df.columns
    ]
    if mental_cols:
        df["mental_health_score"] = df[mental_cols].mean(axis=1)

    if "physical_activity_hours" in df.columns and "screen_time_hours" in df.columns:
        df["lifestyle_balance"] = (
            df["physical_activity_hours"] - df["screen_time_hours"]
        )

    if "daily_sleep_hours" in df.columns:
        df["is_sleep_deprived"] = (df["daily_sleep_hours"] < 6).astype(int)

    if "financial_stress_score" in df.columns:
        if thresholds is not None and "high_fin_stress" in thresholds:
            thr = thresholds["high_fin_stress"]
        else:
            thr = df["financial_stress_score"].quantile(0.75)
        df["is_high_financial_stress"] = (
            df["financial_stress_score"] >= thr
        ).astype(int)
        if thresholds is None:
            thresholds = {}
        thresholds["high_fin_stress"] = thr

    return df


# ── Encoding ───────────────────────────────────────────────────────────────────

def encode_features(
    train: pd.DataFrame,
    val: pd.DataFrame = None,
    test: pd.DataFrame = None,
):
    """
    Fit one-hot encoding and LabelEncoder on **train**, apply to all splits.

    One-hot columns are determined from the train split; val/test are
    reindexed to the same column set (missing columns filled with 0).

    Returns
    -------
    If val and test are provided: (train_enc, val_enc, test_enc, encoders)
    Otherwise: (train_enc, encoders)
    """
    train = train.copy()
    encoders = {}

    cat_present = [c for c in CATEGORICAL_COLS if c in train.columns]

    # Fit one-hot on train
    if cat_present:
        train = pd.get_dummies(train, columns=cat_present, dtype=int)
    train_cols = [c for c in train.columns if c != TARGET_COL]

    # Label-encode target (fit on train)
    le_target = LabelEncoder()
    train[TARGET_COL] = le_target.fit_transform(train[TARGET_COL].astype(str))
    encoders[TARGET_COL] = le_target
    print(f"Target classes: {list(le_target.classes_)}")

    if val is None and test is None:
        # Legacy single-DataFrame mode
        return train, encoders

    def _apply(df):
        df = df.copy()
        cat_cols_here = [c for c in cat_present if c in df.columns]
        if cat_cols_here:
            df = pd.get_dummies(df, columns=cat_cols_here, dtype=int)
        # Align to train columns (handles unseen categories gracefully)
        df = df.reindex(columns=train_cols + [TARGET_COL], fill_value=0)
        df[TARGET_COL] = le_target.transform(df[TARGET_COL].astype(str))
        return df

    val_enc = _apply(val)
    test_enc = _apply(test)
    return train, val_enc, test_enc, encoders


# ── Splitting (raw-data version, legacy kept) ──────────────────────────────────

def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """
    .. deprecated::
        Use ``split_raw()`` instead — it splits before cleaning to eliminate
        leakage of imputation/outlier statistics from val/test into train.

    Kept for backward compatibility with notebooks that work on an already-
    cleaned, already-encoded DataFrame (e.g. quick experiments).
    """
    return split_raw(df, val_size=val_size, test_size=test_size)


# ── Scaling ────────────────────────────────────────────────────────────────────

def scale_features(train, val, test, numeric_cols=None):
    """
    Fit StandardScaler on **train** and transform all three splits.

    Binary (0/1) indicator columns are excluded from scaling.
    """
    if numeric_cols is None:
        numeric_cols = [c for c in NUMERIC_COLS if c in train.columns]

    binary_cols = {
        c for c in numeric_cols if set(train[c].dropna().unique()).issubset({0, 1})
    }
    if binary_cols:
        print(f"Skipping scaling for binary columns: {sorted(binary_cols)}")
    cols_to_scale = [c for c in numeric_cols if c not in binary_cols]

    scaler = StandardScaler()
    train, val, test = train.copy(), val.copy(), test.copy()
    train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
    val[cols_to_scale] = scaler.transform(val[cols_to_scale])
    test[cols_to_scale] = scaler.transform(test[cols_to_scale])
    return train, val, test, scaler
