"""Tests for the donor prediction module."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.predict import (
    EXAMPLE_DONORS,
    RISK_LEVELS,
    format_prediction,
    get_risk_level,
    predict_donor_churn,
)

RESULT_KEYS = {"donor_id", "prediction", "prediction_label", "churn_probability", "risk_level"}


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "Low"),
        (0.2999, "Low"),
        (0.30, "Medium"),
        (0.45, "Medium"),
        (0.5999, "Medium"),
        (0.60, "High"),
        (1.0, "High"),
    ],
)
def test_risk_level_boundaries(probability, expected):
    assert get_risk_level(probability) == expected


@pytest.mark.parametrize("probability", [-0.01, 1.01, math.nan, None])
def test_risk_level_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        get_risk_level(probability)


def test_single_donor_prediction(trained_model, sample_donor):
    result = predict_donor_churn(sample_donor, model=trained_model)
    assert isinstance(result, dict)
    assert set(result) == RESULT_KEYS
    assert result["donor_id"] == sample_donor["donor_id"]
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in RISK_LEVELS


def test_multiple_donor_prediction_from_list(trained_model):
    results = predict_donor_churn(EXAMPLE_DONORS, model=trained_model)
    assert isinstance(results, list)
    assert [r["donor_id"] for r in results] == [d["donor_id"] for d in EXAMPLE_DONORS]
    for result in results:
        assert result["risk_level"] in RISK_LEVELS
        assert result["risk_level"] == get_risk_level(result["churn_probability"])


def test_multiple_donor_prediction_from_dataframe(trained_model, small_dataset):
    batch = small_dataset.head(25)
    results = predict_donor_churn(batch, model=trained_model)
    assert len(results) == 25
    assert all(0.0 <= r["churn_probability"] <= 1.0 for r in results)
    assert {r["risk_level"] for r in results} <= set(RISK_LEVELS)


def test_prediction_loads_model_from_path(saved_model_path, trained_model, sample_donor):
    from_path = predict_donor_churn(sample_donor, model_path=saved_model_path)
    in_memory = predict_donor_churn(sample_donor, model=trained_model)
    assert from_path == in_memory


def test_prediction_matches_probability(trained_model, small_dataset):
    for result in predict_donor_churn(small_dataset.head(50), model=trained_model):
        expected = 1 if result["churn_probability"] > 0.5 else 0
        if abs(result["churn_probability"] - 0.5) > 1e-3:  # ignore rounding right at 0.5
            assert result["prediction"] == expected


def test_prediction_handles_missing_feature_values(trained_model, sample_donor):
    sample_donor.update({"age": None, "email_open_rate": float("nan"), "donation_frequency": None})
    result = predict_donor_churn(sample_donor, model=trained_model)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_prediction_without_donor_id(trained_model, sample_donor):
    sample_donor.pop("donor_id")
    assert predict_donor_churn(sample_donor, model=trained_model)["donor_id"] == "ROW-1"


def test_donor_id_does_not_change_prediction(trained_model, sample_donor):
    first = predict_donor_churn(sample_donor, model=trained_model)
    second = predict_donor_churn({**sample_donor, "donor_id": "SOMEONE-ELSE"}, model=trained_model)
    assert first["churn_probability"] == second["churn_probability"]


def test_missing_feature_column_raises(trained_model, sample_donor):
    sample_donor.pop("days_since_last_donation")
    with pytest.raises(ValueError, match="days_since_last_donation"):
        predict_donor_churn(sample_donor, model=trained_model)


@pytest.mark.parametrize("bad_input", [[], pd.DataFrame(columns=FEATURE_COLUMNS)])
def test_empty_input_raises(trained_model, bad_input):
    with pytest.raises(ValueError):
        predict_donor_churn(bad_input, model=trained_model)


def test_invalid_input_type_raises(trained_model):
    with pytest.raises(TypeError):
        predict_donor_churn("not a donor", model=trained_model)


def test_format_prediction_output(trained_model, sample_donor):
    text = format_prediction(predict_donor_churn(sample_donor, model=trained_model))
    assert f"Donor ID: {sample_donor['donor_id']}" in text
    assert "Prediction: " in text
    assert "Churn Probability: " in text and "%" in text
    assert "Risk Level: " in text
