"""Tests for model training, evaluation and persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.evaluate import (
    compute_metrics,
    evaluate_model,
    get_feature_coefficients,
    positive_class_probability,
    threshold_analysis,
)
from src.generate_dataset import save_dataset
from src.model_io import load_model, save_model
from src.train import build_model_pipeline, run_training, train_model


def test_pipeline_has_expected_configuration():
    pipeline = build_model_pipeline()
    assert isinstance(pipeline, Pipeline)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == 1000
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 42


def test_training_pipeline_can_be_fitted(trained_model):
    assert list(trained_model.classes_) == [0, 1]
    assert hasattr(trained_model.named_steps["classifier"], "coef_")


def test_model_produces_valid_predictions(trained_model, split_data):
    _, X_test, _, _ = split_data
    predictions = trained_model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)) <= {0, 1}


def test_probabilities_are_between_0_and_1(trained_model, split_data):
    _, X_test, _, _ = split_data
    probabilities = trained_model.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_model_beats_random_guessing(trained_model, split_data):
    _, X_test, _, y_test = split_data
    assert roc_auc_score(y_test, positive_class_probability(trained_model, X_test)) > 0.70


def test_model_can_be_saved_and_loaded(trained_model, split_data, tmp_path):
    _, X_test, _, _ = split_data
    path = save_model(trained_model, tmp_path / "nested" / "model.joblib")
    assert path.exists()
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.predict(X_test), trained_model.predict(X_test))
    np.testing.assert_allclose(loaded.predict_proba(X_test), trained_model.predict_proba(X_test))


def test_load_model_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        load_model(tmp_path / "missing.joblib")


def test_train_model_rejects_single_class(split_data):
    X_train, _, y_train, _ = split_data
    with pytest.raises(ValueError, match="both classes"):
        train_model(X_train, y_train * 0)


def test_metrics_are_calculated_from_predictions(trained_model, split_data):
    _, X_test, _, y_test = split_data
    y_pred = trained_model.predict(X_test)
    y_proba = positive_class_probability(trained_model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_test, y_pred))
    assert metrics["precision"] == pytest.approx(precision_score(y_test, y_pred))
    assert metrics["recall"] == pytest.approx(recall_score(y_test, y_pred))
    assert metrics["f1"] == pytest.approx(f1_score(y_test, y_pred))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_test, y_proba))
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(y_test)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0


def test_threshold_analysis_shows_tradeoff(trained_model, split_data):
    _, X_test, _, y_test = split_data
    table = threshold_analysis(y_test, positive_class_probability(trained_model, X_test))
    # Raising the threshold flags fewer donors and can only lower (or keep) recall.
    assert table["donors_flagged"].is_monotonic_decreasing
    assert table["recall"].is_monotonic_decreasing


def test_feature_coefficients_cover_all_features(trained_model):
    table = get_feature_coefficients(trained_model)
    assert len(table) == trained_model.named_steps["classifier"].coef_.shape[1]
    assert "donor_id" not in set(table["feature"])
    days = table.set_index("feature").loc["days_since_last_donation", "coefficient"]
    recurring = table.set_index("feature").loc["recurring_donor", "coefficient"]
    assert days > 0 and recurring < 0  # matches the documented data assumptions


def test_evaluate_model_saves_outputs(trained_model, split_data, tmp_path):
    X_train, X_test, _, y_test = split_data
    info = {"total_donors": len(X_train) + len(X_test), "training_samples": len(X_train),
            "testing_samples": len(X_test), "train_churn_rate": 0.25, "test_churn_rate": 0.25}
    result = evaluate_model(trained_model, X_test, y_test, info, figures_dir=tmp_path / "figures",
                            report_path=tmp_path / "report.txt", metrics_path=tmp_path / "metrics.json", dpi=60)
    for name in ("confusion_matrix.png", "roc_curve.png", "model_metrics_comparison.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 0
    report = (tmp_path / "report.txt").read_text()
    assert f"Accuracy: {result['metrics']['accuracy']:.3f}" in report
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["metrics"]["recall"] == pytest.approx(result["metrics"]["recall"])


def test_run_training_end_to_end(small_dataset, tmp_path):
    data_path = save_dataset(small_dataset, tmp_path / "donors.csv")
    result = run_training(data_path, tmp_path / "model.joblib", tmp_path / "processed")
    assert result["model_path"].exists()
    assert result["train_path"].exists() and result["test_path"].exists()
    info = result["dataset_info"]
    assert info["training_samples"] + info["testing_samples"] == info["total_donors"] == len(small_dataset)
    assert 0.0 <= result["metrics"]["f1"] <= 1.0
