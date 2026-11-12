"""Model evaluation on the untouched test set.

Every number in the report and charts is calculated from real model
predictions - nothing is hard-coded.

Run from the project root (after ``python -m src.train``)::

    python -m src.evaluate
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.config import (
    EVALUATION_REPORT_PATH,
    FIGURE_DPI,
    FIGURES_DIR,
    METRICS_JSON_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.model_io import load_model
from src.preprocess import coerce_feature_types, split_features_target
from src.visualization import COLORS, SEQUENTIAL_CMAP, new_figure, save_figure

CLASS_NAMES: list[str] = ["Active (0)", "Churned (1)"]
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
}
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)


# --------------------------------------------------------------------------- #
# Metric calculations
# --------------------------------------------------------------------------- #
def positive_class_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability of churn (class 1) for each row."""
    churn_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, churn_index]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_proba: Sequence[float]) -> dict:
    """Calculate classification metrics from true labels and model outputs.

    Args:
        y_true: Actual labels (0 = Active, 1 = Churned).
        y_pred: Predicted labels.
        y_proba: Predicted churn probabilities (for ROC-AUC).

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, the confusion
        matrix and the text classification report.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if not len(y_true) == len(y_pred) == len(y_proba):
        raise ValueError("y_true, y_pred and y_proba must have the same length.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "n_samples": int(len(y_true)),
        "n_actual_churned": int(y_true.sum()),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=3, zero_division=0
        ),
    }


def threshold_analysis(
    y_true: Sequence[int], y_proba: Sequence[float], thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Show how precision and recall change with the decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "donors_flagged": int(y_pred.sum()),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def get_feature_coefficients(model: Pipeline) -> pd.DataFrame:
    """Logistic Regression coefficients per (preprocessed) feature.

    Positive coefficient -> higher churn risk; negative -> lower churn risk.
    Numeric features are standardised, so their coefficients are per one
    standard deviation and can be compared with each other.
    """
    names = model.named_steps["preprocessor"].get_feature_names_out()
    names = [name.split("__", 1)[-1] for name in names]
    coefficients = model.named_steps["classifier"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["odds_ratio"] = np.exp(table["coefficient"])
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm: Sequence[Sequence[int]]) -> Figure:
    """Confusion matrix with counts and the share of each actual class."""
    cm = np.asarray(cm)
    row_share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = new_figure(figsize=(7.5, 6.2))
    image = ax.imshow(cm, cmap=SEQUENTIAL_CMAP, vmin=0)
    cells = {(0, 0): "True negative", (0, 1): "False positive", (1, 0): "False negative", (1, 1): "True positive"}
    threshold = cm.max() * 0.55
    for (i, j), name in cells.items():
        color = "white" if cm[i, j] > threshold else COLORS["ink"]
        ax.text(j, i, f"{cm[i, j]:,}\n{row_share[i, j]:.1%} of actual\n{name}", ha="center", va="center",
                color=color, fontsize=11, fontweight="bold" if i == j else "normal")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion matrix (test set)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Number of donors")
    return fig


def plot_roc_curve(y_true: Sequence[int], y_proba: Sequence[float], threshold: float = 0.5) -> Figure:
    """ROC curve with the AUC and the operating point at ``threshold``."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    point_fpr = ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
    point_tpr = ((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)

    fig, ax = new_figure(figsize=(7, 6.2))
    ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=2.2, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.fill_between(fpr, tpr, color=COLORS["primary"], alpha=0.08)
    ax.plot([0, 1], [0, 1], color=COLORS["muted"], linestyle="--", linewidth=1.2, label="Random guessing (AUC = 0.500)")
    ax.scatter([point_fpr], [point_tpr], s=70, color=COLORS["churned"], edgecolor="white", linewidth=1.5, zorder=5,
               label=f"Threshold {threshold:.2f}: recall {point_tpr:.1%}, false-alarm rate {point_fpr:.1%}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (active donors wrongly flagged)")
    ax.set_ylabel("True positive rate / recall (churners caught)")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right", fontsize=9.5)
    return fig


def plot_metrics_comparison(metrics: dict) -> Figure:
    """Bar chart of the actual Accuracy, Precision, Recall and F1-score."""
    labels = list(METRIC_LABELS.values())
    values = [metrics[key] for key in METRIC_LABELS]
    fig, ax = new_figure(figsize=(8.5, 5.5))
    bars = ax.bar(labels, values, color=COLORS["primary"], width=0.6)
    ax.bar_label(bars, labels=[f"{v:.3f}\n({v:.1%})" for v in values], padding=4, fontsize=10.5, color=COLORS["ink"])
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (0 = worst, 1 = best)")
    ax.set_title(f"Model performance on the test set (n = {metrics['n_samples']:,}, synthetic data)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=COLORS["primary"], label="Logistic Regression")], loc="upper right")
    return fig


def plot_feature_coefficients(coefficients: pd.DataFrame) -> Figure:
    """Horizontal bar chart of Logistic Regression coefficients."""
    data = coefficients.sort_values("coefficient")
    colors = [COLORS["churned"] if c > 0 else COLORS["active"] for c in data["coefficient"]]
    fig, ax = new_figure(figsize=(9, 6))
    ax.barh(data["feature"], data["coefficient"], color=colors, height=0.65)
    ax.axvline(0, color=COLORS["ink_secondary"], linewidth=1)
    ax.set_xlabel("Coefficient (log-odds change; numeric features per 1 standard deviation)")
    ax.set_ylabel("Feature")
    ax.set_title("What drives the model's churn predictions?")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Patch(color=COLORS["churned"], label="Increases churn risk"),
            Patch(color=COLORS["active"], label="Decreases churn risk"),
        ],
        loc="lower right",
    )
    return fig


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def format_summary(dataset_info: dict, metrics: dict) -> str:
    """The terminal summary printed after training and evaluation."""
    def fmt(value: float) -> str:
        return f"{value:.3f} ({value:.1%})"

    return "\n".join(
        [
            "DONOR CHURN PREDICTION",
            "======================",
            "",
            "Dataset:",
            f"Total Donors: {dataset_info['total_donors']:,}",
            f"Training Samples: {dataset_info['training_samples']:,}",
            f"Testing Samples: {dataset_info['testing_samples']:,}",
            "",
            "Model:",
            "Logistic Regression",
            "",
            "Evaluation:",
            f"Accuracy: {fmt(metrics['accuracy'])}",
            f"Precision: {fmt(metrics['precision'])}",
            f"Recall: {fmt(metrics['recall'])}",
            f"F1-score: {fmt(metrics['f1'])}",
            f"ROC-AUC: {metrics['roc_auc']:.3f}",
            "",
            "Note: these are results on the held-out TEST set of a SYNTHETIC dataset.",
            "They show how the method works; they do not describe any real organisation.",
        ]
    )


def build_evaluation_report(
    dataset_info: dict, metrics: dict, thresholds: pd.DataFrame, coefficients: pd.DataFrame
) -> str:
    """Full plain-text evaluation report saved to ``reports/model_evaluation.txt``."""
    m = metrics
    sections = [
        format_summary(dataset_info, metrics),
        "",
        "CONFUSION MATRIX (rows = actual, columns = predicted)",
        "-----------------------------------------------------",
        f"{'':<14}{'Pred Active':>14}{'Pred Churned':>14}",
        f"{'Actual Active':<14}{m['true_negatives']:>14,}{m['false_positives']:>14,}",
        f"{'Actual Churned':<14}{m['false_negatives']:>14,}{m['true_positives']:>14,}",
        "",
        f"True negatives  (active, predicted active):   {m['true_negatives']:,}",
        f"False positives (active, flagged as churn):   {m['false_positives']:,}  -> unnecessary retention contact",
        f"False negatives (churned, missed by model):   {m['false_negatives']:,}  -> lost chance to retain donor",
        f"True positives  (churned, correctly flagged): {m['true_positives']:,}",
        "",
        "CLASSIFICATION REPORT",
        "---------------------",
        m["classification_report"].rstrip(),
        "",
        "PRECISION / RECALL TRADE-OFF BY DECISION THRESHOLD",
        "--------------------------------------------------",
        "(the model's default threshold is 0.50)",
        thresholds.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "LOGISTIC REGRESSION COEFFICIENTS (sorted by strength)",
        "-----------------------------------------------------",
        "Positive = increases churn risk, negative = decreases churn risk.",
        coefficients.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "NOTES",
        "-----",
        f"- Test set churn rate: {dataset_info.get('test_churn_rate', float('nan')):.1%}; "
        f"training set churn rate: {dataset_info.get('train_churn_rate', float('nan')):.1%} (stratified split).",
        "- class_weight='balanced' makes the model pay more attention to churners. This raises recall",
        "  but also shifts predicted probabilities upwards, so treat them as risk scores, not calibrated",
        "  real-world probabilities.",
        "- All results come from synthetic data generated by src/generate_dataset.py.",
    ]
    return "\n".join(sections) + "\n"


def _json_ready(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "classification_report"}


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_info: dict,
    figures_dir: str | Path = FIGURES_DIR,
    report_path: str | Path = EVALUATION_REPORT_PATH,
    metrics_path: str | Path = METRICS_JSON_PATH,
    dpi: int = FIGURE_DPI,
) -> dict:
    """Evaluate ``model`` on the test set and save figures, report and metrics.

    Returns:
        Dictionary with ``metrics``, ``thresholds``, ``coefficients``,
        ``figures`` (name -> path), ``report_path`` and ``metrics_path``.
    """
    figures_dir, report_path, metrics_path = Path(figures_dir), Path(report_path), Path(metrics_path)
    y_pred = model.predict(X_test)
    y_proba = positive_class_probability(model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    thresholds = threshold_analysis(y_test, y_proba)
    coefficients = get_feature_coefficients(model)

    figures = {
        "confusion_matrix.png": plot_confusion_matrix(metrics["confusion_matrix"]),
        "roc_curve.png": plot_roc_curve(y_test, y_proba),
        "model_metrics_comparison.png": plot_metrics_comparison(metrics),
        "feature_coefficients.png": plot_feature_coefficients(coefficients),
    }
    saved = {name: save_figure(fig, figures_dir / name, dpi=dpi) for name, fig in figures.items()}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_evaluation_report(dataset_info, metrics, thresholds, coefficients), encoding="utf-8")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"dataset": dataset_info, "metrics": _json_ready(metrics)}, indent=2),
                            encoding="utf-8")
    return {
        "metrics": metrics,
        "thresholds": thresholds,
        "coefficients": coefficients,
        "figures": saved,
        "report_path": report_path,
        "metrics_path": metrics_path,
    }


def load_split(path: str | Path) -> pd.DataFrame:
    """Load a saved train/test CSV created by ``python -m src.train``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found. Train the model first with: python -m src.train")
    return coerce_feature_types(pd.read_csv(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained donor churn model on the test set.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to the saved model.")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA_PATH, help="Saved training split CSV.")
    parser.add_argument("--test-data", type=Path, default=TEST_DATA_PATH, help="Saved test split CSV.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Folder for PNG charts.")
    parser.add_argument("--report", type=Path, default=EVALUATION_REPORT_PATH, help="Text report path.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    model = load_model(args.model)
    train_df = load_split(args.train_data)
    test_df = load_split(args.test_data)
    X_test, y_test = split_features_target(test_df)
    dataset_info = {
        "total_donors": len(train_df) + len(test_df),
        "training_samples": len(train_df),
        "testing_samples": len(test_df),
        "train_churn_rate": float(train_df[TARGET_COLUMN].mean()),
        "test_churn_rate": float(test_df[TARGET_COLUMN].mean()),
    }
    result = evaluate_model(model, X_test, y_test, dataset_info, args.figures_dir, args.report)
    print(format_summary(dataset_info, result["metrics"]))
    print("\nClassification report:")
    print(result["metrics"]["classification_report"])
    print(f"Report saved to:  {result['report_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print("Figures saved:")
    for path in result["figures"].values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
