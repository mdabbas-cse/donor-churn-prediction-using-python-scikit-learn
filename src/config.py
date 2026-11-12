"""Central configuration: file paths, column names and shared constants.

Every path is built from the project root, so all commands work when they are
run from the project root (for example ``python -m src.train``) no matter where
the repository lives on disk.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_DATA_PATH: Path = RAW_DATA_DIR / "donors.csv"
CLEAN_DATA_PATH: Path = PROCESSED_DATA_DIR / "donors_clean.csv"
TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "donor_churn_model.joblib"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
EVALUATION_REPORT_PATH: Path = REPORTS_DIR / "model_evaluation.txt"
METRICS_JSON_PATH: Path = REPORTS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Reproducibility and data split
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
DEFAULT_N_DONORS: int = 5_000

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ID_COLUMN: str = "donor_id"
TARGET_COLUMN: str = "churn"

# Continuous numeric features -> median imputation + standard scaling.
NUMERIC_FEATURES: list[str] = [
    "age",
    "donation_count",
    "total_donated",
    "avg_donation",
    "days_since_last_donation",
    "donor_tenure_months",
    "email_open_rate",
    "campaign_response_rate",
]

# 0/1 flags -> most-frequent imputation, no scaling needed.
BINARY_FEATURES: list[str] = [
    "recurring_donor",
    "previous_churn_flag",
]

# Text categories -> most-frequent imputation + one-hot encoding.
CATEGORICAL_FEATURES: list[str] = [
    "donation_frequency",
]

# The model only ever sees these columns. donor_id and churn are excluded.
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Column order of the generated CSV file.
DATASET_COLUMNS: list[str] = [
    ID_COLUMN,
    "age",
    "donation_count",
    "total_donated",
    "avg_donation",
    "days_since_last_donation",
    "donation_frequency",
    "donor_tenure_months",
    "email_open_rate",
    "campaign_response_rate",
    "recurring_donor",
    "previous_churn_flag",
    TARGET_COLUMN,
]

DONATION_FREQUENCIES: tuple[str, ...] = ("Monthly", "Quarterly", "Yearly")
CLASS_LABELS: dict[int, str] = {0: "Active", 1: "Churned"}

# --------------------------------------------------------------------------- #
# Prediction risk bands (illustrative only, not validated fundraising policy)
# --------------------------------------------------------------------------- #
LOW_RISK_UPPER: float = 0.30  # probability < 0.30          -> Low
MEDIUM_RISK_UPPER: float = 0.60  # 0.30 <= probability < 0.60 -> Medium
# probability >= 0.60                                        -> High

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
FIGURE_DPI: int = 300
