"""Saving and loading the trained model pipeline with Joblib."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH


def save_model(model: Pipeline, path: str | Path = MODEL_PATH) -> Path:
    """Save a fitted pipeline (preprocessing + classifier) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    """Load a fitted pipeline from ``path``.

    Raises:
        FileNotFoundError: If the model file does not exist yet.
        TypeError: If the file does not contain a scikit-learn Pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Train the model first with: python -m src.train"
        )
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a scikit-learn Pipeline in '{path}', got {type(model).__name__}.")
    return model
