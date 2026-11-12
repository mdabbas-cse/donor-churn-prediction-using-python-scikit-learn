"""Tests for validation, cleaning and the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BINARY_FEATURES,
    DONATION_FREQUENCIES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.preprocess import (
    build_preprocessor,
    clean_dataset,
    load_dataset,
    split_features_target,
    split_train_test,
    validate_dataset,
)


def test_preprocessor_handles_missing_values(small_dataset):
    X, _ = split_features_target(clean_dataset(small_dataset))
    X = X.copy()
    X.loc[X.index[:25], "age"] = np.nan
    X.loc[X.index[:25], "total_donated"] = np.nan
    X.loc[X.index[:25], "recurring_donor"] = np.nan
    X.loc[X.index[:25], "donation_frequency"] = np.nan
    assert X.isna().any().any()

    transformed = build_preprocessor().fit_transform(X)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_preprocessor_output_columns(small_dataset):
    X, _ = split_features_target(clean_dataset(small_dataset))
    preprocessor = build_preprocessor().fit(X)
    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    expected_width = len(NUMERIC_FEATURES) + len(BINARY_FEATURES) + len(DONATION_FREQUENCIES)
    assert len(names) == expected_width
    assert ID_COLUMN not in names
    assert TARGET_COLUMN not in names


def test_unknown_category_is_ignored(small_dataset):
    X, _ = split_features_target(clean_dataset(small_dataset))
    preprocessor = build_preprocessor().fit(X)
    new_row = X.iloc[[0]].copy()
    new_row["donation_frequency"] = "Weekly"  # never seen during training
    assert not np.isnan(preprocessor.transform(new_row)).any()


def test_split_features_target_excludes_id_and_target(small_dataset):
    X, y = split_features_target(clean_dataset(small_dataset))
    assert list(X.columns) == FEATURE_COLUMNS
    assert ID_COLUMN not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_split_is_80_20_and_stratified(small_dataset):
    X, y = split_features_target(clean_dataset(small_dataset))
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    assert len(X_train) == int(len(X) * 0.8)
    assert len(X_test) == len(X) - len(X_train)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_preprocessing_is_fitted_on_training_data_only(trained_model, split_data):
    """The scaler must learn its statistics from training rows only (no leakage)."""
    X_train, X_test, _, _ = split_data
    scaler = trained_model.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    imputer = trained_model.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    train_imputed = X_train[NUMERIC_FEATURES].fillna(X_train[NUMERIC_FEATURES].median())
    np.testing.assert_allclose(scaler.mean_, train_imputed.mean().to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(imputer.statistics_, X_train[NUMERIC_FEATURES].median().to_numpy(), rtol=1e-6)
    all_rows = pd.concat([X_train, X_test])[NUMERIC_FEATURES]
    all_imputed = all_rows.fillna(all_rows.median())
    assert not np.allclose(scaler.mean_, all_imputed.mean().to_numpy())


def test_validate_dataset_reports_missing_and_duplicates(small_dataset):
    df = pd.concat([small_dataset, small_dataset.iloc[[0, 1]]], ignore_index=True)
    report = validate_dataset(df)
    assert report.duplicate_rows == 2
    assert report.duplicate_ids == 2
    assert report.total_missing == int(df.isna().sum().sum())
    assert report.missing_values["age"] > 0
    assert "Duplicate rows: 2" in report.to_text()


def test_validate_dataset_rejects_missing_columns(small_dataset):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(small_dataset.drop(columns=["email_open_rate"]))


def test_validate_dataset_rejects_invalid_target(small_dataset):
    df = small_dataset.copy()
    df.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="must only contain 0 and 1"):
        validate_dataset(df)


def test_validate_dataset_flags_wrong_types(small_dataset):
    df = small_dataset.copy()
    df["age"] = df["age"].astype(str)
    report = validate_dataset(df)
    assert any("'age' should be numeric" in issue for issue in report.issues)


def test_clean_dataset_removes_duplicates_and_fixes_types(small_dataset):
    df = pd.concat([small_dataset, small_dataset.iloc[[0, 1, 2]]], ignore_index=True)
    df["donation_frequency"] = df["donation_frequency"].astype(object)
    df["age"] = df["age"].astype(object)  # allow an invalid text value, as in a messy CSV
    df.loc[3, "donation_frequency"] = "  monthly "
    df.loc[4, "age"] = "not a number"
    df.loc[5, TARGET_COLUMN] = np.nan
    cleaned = clean_dataset(df)
    assert cleaned[ID_COLUMN].is_unique
    assert len(cleaned) == len(small_dataset) - 1  # duplicates and the unlabeled row removed
    assert cleaned.loc[cleaned[ID_COLUMN] == df.loc[3, ID_COLUMN], "donation_frequency"].item() == "Monthly"
    assert pd.isna(cleaned.loc[cleaned[ID_COLUMN] == df.loc[4, ID_COLUMN], "age"].item())
    assert cleaned[TARGET_COLUMN].dtype.kind == "i"


def test_load_dataset_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.generate_dataset"):
        load_dataset(tmp_path / "does_not_exist.csv")
