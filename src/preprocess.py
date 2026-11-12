"""Data loading, validation, cleaning and the Scikit-learn preprocessing pipeline.

Key idea - preventing data leakage
----------------------------------
The :func:`build_preprocessor` function only *describes* the preprocessing
steps. Nothing is learned until ``fit`` is called. Because the preprocessor is
placed inside the model :class:`~sklearn.pipeline.Pipeline` and fitted on the
training data only, the medians, means, standard deviations and categories are
learned from training rows only. The test rows are never used for fitting.

Run from the project root to see a data-quality report::

    python -m src.preprocess
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    DONATION_FREQUENCIES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the donor CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Generate it first with: python -m src.generate_dataset"
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at '{path}' is empty.")
    return df


@dataclass
class DataQualityReport:
    """Results of the basic data-quality checks."""

    n_rows: int
    n_columns: int
    missing_values: dict[str, int]
    duplicate_rows: int
    duplicate_ids: int
    issues: list[str] = field(default_factory=list)

    @property
    def total_missing(self) -> int:
        """Total number of missing cells."""
        return int(sum(self.missing_values.values()))

    def to_text(self) -> str:
        """Format the report for printing."""
        lines = [
            f"Rows: {self.n_rows:,} | Columns: {self.n_columns}",
            f"Duplicate rows: {self.duplicate_rows}",
            f"Duplicate donor IDs: {self.duplicate_ids}",
            f"Missing values: {self.total_missing:,} cells",
        ]
        for column, count in self.missing_values.items():
            if count:
                lines.append(f"  - {column}: {count:,} ({count / self.n_rows:.1%})")
        if self.issues:
            lines.append("Data type / value issues:")
            lines.extend(f"  - {issue}" for issue in self.issues)
        else:
            lines.append("Data types: OK")
        return "\n".join(lines)


def validate_dataset(df: pd.DataFrame, require_target: bool = True) -> DataQualityReport:
    """Check required columns, missing values, duplicates and data types.

    Args:
        df: Raw donor data.
        require_target: Whether the ``churn`` column must be present.

    Returns:
        A :class:`DataQualityReport`. Soft problems (for example an unexpected
        category) are listed in ``report.issues``.

    Raises:
        ValueError: If required columns are missing or the target contains
            values other than 0 and 1.
    """
    required = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    issues: list[str] = []
    for column in NUMERIC_FEATURES + BINARY_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            issues.append(f"'{column}' should be numeric but has dtype '{df[column].dtype}'.")
    for column in BINARY_FEATURES:
        values = set(pd.to_numeric(df[column], errors="coerce").dropna().unique())
        if not values <= {0, 1}:
            issues.append(f"'{column}' should only contain 0/1, found {sorted(values)}.")
    unknown_categories = set(df["donation_frequency"].dropna().unique()) - set(DONATION_FREQUENCIES)
    if unknown_categories:
        issues.append(f"'donation_frequency' has unexpected values: {sorted(unknown_categories)}.")

    if require_target:
        target_values = set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().unique())
        if not target_values <= {0, 1}:
            raise ValueError(f"Target '{TARGET_COLUMN}' must only contain 0 and 1, found {sorted(target_values)}.")
        if df[TARGET_COLUMN].isna().any():
            issues.append(f"'{TARGET_COLUMN}' has {int(df[TARGET_COLUMN].isna().sum())} missing values.")

    duplicate_ids = int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else 0
    return DataQualityReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing_values={column: int(count) for column, count in df.isna().sum().items()},
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_ids=duplicate_ids,
        issues=issues,
    )


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def coerce_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric features as floats and categories as tidy text.

    Invalid numbers become ``NaN`` and blank categories become missing, so the
    imputers in the pipeline can handle them.
    """
    df = df.copy()
    for column in NUMERIC_FEATURES + BINARY_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            cleaned = df[column].astype(object).where(df[column].notna(), np.nan)
            cleaned = cleaned.map(lambda v: v.strip().title() if isinstance(v, str) else v)
            df[column] = cleaned.replace("", np.nan).astype(object)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, remove duplicates and drop rows without a target label.

    Missing *feature* values are deliberately kept: they are filled later by
    the imputers inside the pipeline, which learn from training data only.
    """
    df = coerce_feature_types(df)
    df = df.drop_duplicates()
    if ID_COLUMN in df.columns:
        df = df.drop_duplicates(subset=ID_COLUMN, keep="first")
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the target (y).

    ``donor_id`` is an identifier, not a behaviour, so it is excluded from X.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split (80/20 by default).

    ``stratify=y`` keeps the churn percentage the same in both parts, which
    matters when one class (churned donors) is much smaller than the other.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) preprocessing step.

    * Numeric features: fill gaps with the training median, then standardise
      (mean 0, standard deviation 1).
    * Binary 0/1 features: fill gaps with the most frequent value.
    * Categorical features: fill gaps with the most frequent category, then
      one-hot encode. Unknown categories at prediction time are ignored.
    * Any other column (such as ``donor_id``) is dropped.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and clean the donor dataset.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Raw dataset CSV path.")
    parser.add_argument("--output", type=Path, default=CLEAN_DATA_PATH, help="Cleaned CSV output path.")
    return parser.parse_args()


def main() -> None:
    """Print a data-quality report and save a cleaned copy of the data."""
    args = _parse_args()
    raw = load_dataset(args.data)
    print("DATA QUALITY REPORT (raw data)")
    print("==============================")
    print(validate_dataset(raw).to_text())

    clean = clean_dataset(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    X, y = split_features_target(clean)
    X_train, X_test, _, _ = split_train_test(X, y)
    print(f"\nRows after cleaning: {len(clean):,} (removed {len(raw) - len(clean):,})")
    print(f"Features used by the model ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")
    print(f"Train/test split: {len(X_train):,} / {len(X_test):,} rows (stratified)")
    print("Missing feature values are kept and filled by the pipeline imputers during training.")
    print(f"Cleaned data saved to: {args.output}")


if __name__ == "__main__":
    main()
