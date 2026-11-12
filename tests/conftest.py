"""Shared pytest fixtures.

The fixtures build a small synthetic dataset and train a temporary model, so the
tests never depend on files created by running the scripts manually.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.generate_dataset import generate_donor_dataset
from src.model_io import save_model
from src.predict import EXAMPLE_DONORS
from src.preprocess import clean_dataset, split_features_target, split_train_test
from src.train import train_model

SMALL_N_DONORS = 800


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    """A small but realistic dataset (with missing values) for fast tests."""
    return generate_donor_dataset(n_donors=SMALL_N_DONORS, random_state=7)


@pytest.fixture(scope="session")
def split_data(small_dataset: pd.DataFrame):
    """Stratified train/test split of the small dataset."""
    X, y = split_features_target(clean_dataset(small_dataset))
    return split_train_test(X, y)


@pytest.fixture(scope="session")
def trained_model(split_data):
    """A pipeline trained on the small training split only."""
    X_train, _, y_train, _ = split_data
    return train_model(X_train, y_train)


@pytest.fixture(scope="session")
def saved_model_path(trained_model, tmp_path_factory) -> Path:
    """The trained model saved to a temporary folder."""
    return save_model(trained_model, tmp_path_factory.mktemp("models") / "model.joblib")


@pytest.fixture
def sample_donor() -> dict:
    """One complete donor record (a fresh copy for every test)."""
    return dict(EXAMPLE_DONORS[0])
