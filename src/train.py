"""Train the Logistic Regression donor churn model.

Workflow:
1. Load and validate the raw dataset.
2. Clean it (types, duplicates, missing target).
3. Split into 80% training and 20% test data (stratified by churn).
4. Fit ONE pipeline (preprocessing + Logistic Regression) on training data only.
5. Save the fitted pipeline with Joblib and the splits to ``data/processed``.
6. Print a summary with test-set metrics.

Run from the project root::

    python -m src.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_PATH
from src.evaluate import compute_metrics, format_summary, positive_class_probability
from src.model_io import save_model
from src.preprocess import (
    build_preprocessor,
    clean_dataset,
    load_dataset,
    split_features_target,
    split_train_test,
    validate_dataset,
)


def build_model_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create the full (unfitted) pipeline: preprocessing -> Logistic Regression.

    ``class_weight="balanced"`` gives churned donors (the smaller class) more
    weight during training so the model does not simply predict "Active"
    for everyone.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            ),
        ]
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the pipeline on training data only (prevents data leakage)."""
    if X_train.empty:
        raise ValueError("Training data is empty.")
    if pd.Series(y_train).nunique() < 2:
        raise ValueError("Training data must contain both classes (0 = Active, 1 = Churned).")
    model = build_model_pipeline()
    model.fit(X_train, y_train)
    return model


def run_training(
    data_path: str | Path = RAW_DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> dict:
    """Run the full training workflow and return its results.

    Returns:
        Dictionary with ``model``, ``metrics`` (test set), ``dataset_info``,
        ``quality_report``, ``model_path``, ``train_path`` and ``test_path``.
    """
    processed_dir = Path(processed_dir)
    raw = load_dataset(data_path)
    quality_report = validate_dataset(raw)
    df = clean_dataset(raw)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Keep the exact split on disk so evaluation uses the same untouched test rows.
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    df.loc[X_train.index].to_csv(train_path, index=False)
    df.loc[X_test.index].to_csv(test_path, index=False)

    model = train_model(X_train, y_train)
    saved_path = save_model(model, model_path)

    # The test set is only used here to *report* performance, never to tune the model.
    metrics = compute_metrics(y_test, model.predict(X_test), positive_class_probability(model, X_test))
    dataset_info = {
        "total_donors": len(df),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
    }
    return {
        "model": model,
        "metrics": metrics,
        "dataset_info": dataset_info,
        "quality_report": quality_report,
        "model_path": saved_path,
        "train_path": train_path,
        "test_path": test_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the donor churn Logistic Regression model.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Raw dataset CSV path.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Where to save the trained model.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR, help="Folder for train/test CSVs.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    result = run_training(args.data, args.model, args.processed_dir)
    info = result["dataset_info"]

    print("DATA QUALITY")
    print("============")
    print(result["quality_report"].to_text())
    print(
        f"\nStratified split check -> churn rate train: {info['train_churn_rate']:.1%}, "
        f"test: {info['test_churn_rate']:.1%}\n"
    )
    print(format_summary(info, result["metrics"]))
    print(f"\nModel saved to: {result['model_path']}")
    print(f"Train/test splits saved to: {result['train_path'].parent}")
    print("Next step: python -m src.evaluate  (saves figures and the full evaluation report)")


if __name__ == "__main__":
    main()
