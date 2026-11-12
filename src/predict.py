"""Predict churn risk for one or many donors with the saved model.

Risk levels (illustrative thresholds, NOT validated fundraising policy):

* Low Risk:    churn probability below 30%
* Medium Risk: 30% to below 60%
* High Risk:   60% or higher

Run from the project root (after ``python -m src.train``)::

    python -m src.predict                                   # built-in example donors
    python -m src.predict --input data/processed/test.csv   # score a CSV file
    python -m src.predict --input my_donors.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, ID_COLUMN, LOW_RISK_UPPER, MEDIUM_RISK_UPPER, MODEL_PATH
from src.evaluate import positive_class_probability
from src.model_io import load_model
from src.preprocess import coerce_feature_types

RISK_LEVELS: tuple[str, ...] = ("Low", "Medium", "High")
PREDICTION_LABELS: dict[int, str] = {0: "Likely to Stay", 1: "Churn Risk"}

DonorRecord = Mapping[str, Any]
DonorInput = DonorRecord | Sequence[DonorRecord] | pd.DataFrame

# Hypothetical new donors used by the command-line demo (not from the dataset).
EXAMPLE_DONORS: list[dict[str, Any]] = [
    {
        "donor_id": "D1001",
        "age": 34,
        "donation_count": 3,
        "total_donated": 120.0,
        "avg_donation": 40.0,
        "days_since_last_donation": 420,
        "donation_frequency": "Yearly",
        "donor_tenure_months": 30,
        "email_open_rate": 9.5,
        "campaign_response_rate": 3.0,
        "recurring_donor": 0,
        "previous_churn_flag": 1,
    },
    {
        "donor_id": "D1002",
        "age": 45,
        "donation_count": 8,
        "total_donated": 480.0,
        "avg_donation": 60.0,
        "days_since_last_donation": 130,
        "donation_frequency": "Quarterly",
        "donor_tenure_months": 26,
        "email_open_rate": 35.0,
        "campaign_response_rate": 15.0,
        "recurring_donor": 0,
        "previous_churn_flag": 0,
    },
    {
        "donor_id": "D1003",
        "age": 58,
        "donation_count": 62,
        "total_donated": 1860.0,
        "avg_donation": 30.0,
        "days_since_last_donation": 12,
        "donation_frequency": "Monthly",
        "donor_tenure_months": 64,
        "email_open_rate": 68.0,
        "campaign_response_rate": 34.0,
        "recurring_donor": 1,
        "previous_churn_flag": 0,
    },
]


def get_risk_level(probability: float) -> str:
    """Map a churn probability (0-1) to 'Low', 'Medium' or 'High'.

    Raises:
        ValueError: If ``probability`` is not a number between 0 and 1.
    """
    if probability is None or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability!r}.")
    if probability < LOW_RISK_UPPER:
        return "Low"
    if probability < MEDIUM_RISK_UPPER:
        return "Medium"
    return "High"


def _to_dataframe(donor_data: DonorInput) -> tuple[pd.DataFrame, bool]:
    """Convert supported inputs into a DataFrame. Returns ``(df, is_single)``."""
    if isinstance(donor_data, pd.DataFrame):
        return donor_data.reset_index(drop=True), False
    if isinstance(donor_data, Mapping):
        return pd.DataFrame([dict(donor_data)]), True
    if isinstance(donor_data, Sequence) and not isinstance(donor_data, (str, bytes)):
        if not all(isinstance(row, Mapping) for row in donor_data):
            raise TypeError("A list of donors must contain dictionaries.")
        return pd.DataFrame([dict(row) for row in donor_data]), False
    raise TypeError(
        "donor_data must be a dict (one donor), a list of dicts or a pandas DataFrame, "
        f"got {type(donor_data).__name__}."
    )


def predict_donor_churn(
    donor_data: DonorInput,
    model: Pipeline | None = None,
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict churn for one donor or many donors.

    Args:
        donor_data: One donor as a ``dict``, or several donors as a list of
            dicts or a DataFrame. Each donor needs every feature column listed
            in ``src.config.FEATURE_COLUMNS``; ``donor_id`` is optional and is
            never used by the model. Missing feature *values* (``None``/NaN) are
            allowed and are filled by the pipeline's imputers.
        model: An already loaded pipeline. If ``None`` it is loaded from ``model_path``.
        model_path: Location of the saved Joblib model.

    Returns:
        For a single dict: one result dict. For a list/DataFrame: a list of
        result dicts. Each result contains ``donor_id``, ``prediction``
        (1 = churn, 0 = stay), ``prediction_label``, ``churn_probability``
        (0-1) and ``risk_level``.
    """
    df, is_single = _to_dataframe(donor_data)
    if df.empty:
        raise ValueError("No donor records were provided.")
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Donor data is missing required feature columns: {missing}")

    if model is None:
        model = load_model(model_path)

    features = coerce_feature_types(df[FEATURE_COLUMNS])
    predictions = model.predict(features).astype(int)
    probabilities = positive_class_probability(model, features)

    if ID_COLUMN in df.columns:
        ids = [str(v) if pd.notna(v) else f"ROW-{i + 1}" for i, v in enumerate(df[ID_COLUMN])]
    else:
        ids = [f"ROW-{i + 1}" for i in range(len(df))]

    results = []
    for donor_id, prediction, probability in zip(ids, predictions, probabilities):
        probability = round(float(probability), 4)
        results.append(
            {
                "donor_id": donor_id,
                "prediction": int(prediction),
                "prediction_label": PREDICTION_LABELS[int(prediction)],
                "churn_probability": probability,
                "risk_level": get_risk_level(probability),
            }
        )
    return results[0] if is_single else results


def format_prediction(result: Mapping[str, Any]) -> str:
    """Format one prediction result for printing."""
    return "\n".join(
        [
            f"Donor ID: {result['donor_id']}",
            f"Prediction: {result['prediction_label']}",
            f"Churn Probability: {result['churn_probability']:.1%}",
            f"Risk Level: {result['risk_level']}",
        ]
    )


def print_predictions(results: dict[str, Any] | list[dict[str, Any]], max_rows: int = 20) -> None:
    """Print one or more prediction results in a readable layout."""
    results = [results] if isinstance(results, Mapping) else list(results)
    for result in results[:max_rows]:
        print(format_prediction(result))
        print("-" * 32)
    if len(results) > max_rows:
        print(f"... {len(results) - max_rows:,} more donors not shown.")


def _risk_summary(results: list[dict[str, Any]]) -> str:
    counts = pd.Series([r["risk_level"] for r in results]).value_counts().reindex(RISK_LEVELS, fill_value=0)
    return " | ".join(f"{level}: {count:,}" for level, count in counts.items())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict donor churn risk with the trained model.")
    parser.add_argument("--input", type=Path, help="CSV file with donor features (optional).")
    parser.add_argument("--output", type=Path, help="Where to save predictions as CSV (optional).")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to the saved model.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    model = load_model(args.model)

    print("DONOR CHURN PREDICTION RESULTS")
    print("==============================")
    if args.input is None:
        print("\n[1] Single donor prediction\n")
        print_predictions(predict_donor_churn(EXAMPLE_DONORS[0], model=model))
        print("\n[2] Batch prediction for multiple donors\n")
        results = predict_donor_churn(EXAMPLE_DONORS, model=model)
    else:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file '{args.input}' does not exist.")
        print(f"\nScoring donors from: {args.input}\n")
        results = predict_donor_churn(pd.read_csv(args.input), model=model)
    print_predictions(results)
    print(f"Risk level summary -> {_risk_summary(results)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")

    print(
        "\nNote: risk thresholds (30% / 60%) are illustrative, not validated fundraising policy. "
        "Probabilities come from a model trained on synthetic data with class_weight='balanced', "
        "so treat them as relative risk scores."
    )


if __name__ == "__main__":
    main()
