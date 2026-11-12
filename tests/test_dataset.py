"""Tests for synthetic dataset generation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATASET_COLUMNS, DONATION_FREQUENCIES, FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from src.generate_dataset import generate_donor_dataset, save_dataset


@pytest.fixture(scope="module")
def full_dataset() -> pd.DataFrame:
    """The default-size dataset (generation takes well under a second)."""
    return generate_donor_dataset()


def test_creates_at_least_5000_records(full_dataset):
    assert len(full_dataset) >= 5_000


def test_donor_ids_are_unique(full_dataset):
    assert full_dataset[ID_COLUMN].is_unique


def test_no_duplicate_donor_ids_after_saving(full_dataset, tmp_path):
    path = save_dataset(full_dataset, tmp_path / "raw" / "donors.csv")
    reloaded = pd.read_csv(path)
    assert path.exists()
    assert len(reloaded) == len(full_dataset)
    assert reloaded[ID_COLUMN].duplicated().sum() == 0


def test_target_contains_only_0_and_1(full_dataset):
    assert set(full_dataset[TARGET_COLUMN].unique()) == {0, 1}


def test_required_columns_exist(full_dataset):
    assert list(full_dataset.columns) == DATASET_COLUMNS
    for column in FEATURE_COLUMNS + [ID_COLUMN, TARGET_COLUMN]:
        assert column in full_dataset.columns


def test_generation_is_reproducible():
    first = generate_donor_dataset(n_donors=300, random_state=123)
    second = generate_donor_dataset(n_donors=300, random_state=123)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_data():
    first = generate_donor_dataset(n_donors=300, random_state=1)
    second = generate_donor_dataset(n_donors=300, random_state=2)
    assert not first.equals(second)


def test_class_imbalance_is_realistic(full_dataset):
    churn_rate = full_dataset[TARGET_COLUMN].mean()
    assert 0.15 < churn_rate < 0.40


def test_value_ranges_are_valid(full_dataset):
    df = full_dataset
    assert df["age"].dropna().between(18, 90).all()
    assert (df["donation_count"] >= 1).all()
    assert (df["total_donated"] > 0).all()
    assert df["email_open_rate"].dropna().between(0, 100).all()
    assert df["campaign_response_rate"].dropna().between(0, 100).all()
    assert set(df["recurring_donor"].unique()) <= {0, 1}
    assert set(df["previous_churn_flag"].unique()) <= {0, 1}
    assert set(df["donation_frequency"].dropna().unique()) <= set(DONATION_FREQUENCIES)
    # A donor cannot have last given before the relationship started.
    assert (df["days_since_last_donation"] <= df["donor_tenure_months"] * 30).all()


def test_average_donation_is_consistent(full_dataset):
    df = full_dataset
    implied = df["total_donated"] / df["donation_count"]
    assert (implied - df["avg_donation"]).abs().max() < 0.01


def test_feature_relationships_follow_assumptions(full_dataset):
    df = full_dataset
    churned, active = df[df[TARGET_COLUMN] == 1], df[df[TARGET_COLUMN] == 0]
    assert churned["days_since_last_donation"].mean() > active["days_since_last_donation"].mean()
    assert churned["email_open_rate"].mean() < active["email_open_rate"].mean()
    assert churned["campaign_response_rate"].mean() < active["campaign_response_rate"].mean()
    rate_by_recurring = df.groupby("recurring_donor")[TARGET_COLUMN].mean()
    assert rate_by_recurring[1] < rate_by_recurring[0]


def test_dataset_contains_some_missing_values(full_dataset):
    assert full_dataset[["age", "email_open_rate", "donation_frequency"]].isna().any().all()
    assert full_dataset[TARGET_COLUMN].notna().all()
    assert full_dataset[ID_COLUMN].notna().all()


@pytest.mark.parametrize(
    "kwargs",
    [{"n_donors": 0}, {"target_churn_rate": 0.0}, {"target_churn_rate": 1.5}, {"missing_rate": 0.7}],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        generate_donor_dataset(**kwargs)
