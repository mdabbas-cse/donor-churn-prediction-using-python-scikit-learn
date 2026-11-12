"""Generate a reproducible, synthetic donor dataset for churn prediction.

The data is completely artificial. It is *inspired by* nonprofit fundraising
(for example organisations such as UNICEF) but it is NOT real donor data and
does not describe any real organisation or person.

Labelling assumptions (how ``churn`` is created)
------------------------------------------------
1. **Snapshot design (prevents target leakage).** Every feature describes a
   donor *as of one observation date*. ``churn`` describes what happens *after*
   that date: ``1`` means the donor makes no donation in the following
   12 months. No feature contains information from that future window.
2. **Hidden engagement.** Each donor has an unobserved "engagement" score. It
   drives email opens, campaign responses, how often they give and how
   recently they gave, so these features are correlated - as in real life.
3. **True churn risk** is a logistic function of the features below. The
   direction of every effect follows common fundraising experience:

   ===============================  ==========================================
   Driver                           Effect on churn risk
   ===============================  ==========================================
   days_since_last_donation (log)   strong increase (the longer the gap)
   email_open_rate                  decrease (more engaged readers stay)
   campaign_response_rate           decrease
   recurring_donor = 1              strong decrease (automatic gifts)
   donor_tenure_months (log)        decrease (long relationships last)
   previous_churn_flag = 1          increase (lapsed before -> may lapse again)
   age                              small decrease
   hidden engagement                decrease (not visible to the model)
   random "life event" noise        unpredictable (not visible to the model)
   ===============================  ==========================================

4. **No perfect prediction.** The label is *sampled* from the risk
   probability (a coin flip weighted by the risk), and two drivers are hidden
   from the model. Donors with identical features can therefore have
   different outcomes, which creates realistic class overlap.
5. **Class imbalance.** The intercept is calibrated so roughly 25% of donors
   churn (configurable with ``target_churn_rate``).
6. **Messy data.** A small share of ``age``, ``email_open_rate``,
   ``campaign_response_rate`` and ``donation_frequency`` values are removed at
   random (after labelling) so the preprocessing pipeline has missing values
   to handle.

Run from the project root::

    python -m src.generate_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATASET_COLUMNS,
    DEFAULT_N_DONORS,
    DONATION_FREQUENCIES,
    ID_COLUMN,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

# Share of donors per giving frequency.
FREQUENCY_PROBABILITIES: tuple[float, ...] = (0.35, 0.30, 0.35)
# Probability that a donor gives through an automatic recurring plan.
RECURRING_PROBABILITY: dict[str, float] = {"Monthly": 0.75, "Quarterly": 0.40, "Yearly": 0.12}
# Expected number of gifts per month for each frequency.
GIFTS_PER_MONTH: dict[str, float] = {"Monthly": 1.0, "Quarterly": 1 / 3, "Yearly": 1 / 12}
# Typical number of days between two gifts.
TYPICAL_GAP_DAYS: dict[str, float] = {"Monthly": 30.0, "Quarterly": 91.0, "Yearly": 365.0}
# Typical gift size in dollars (frequent donors give smaller amounts).
TYPICAL_GIFT: dict[str, float] = {"Monthly": 25.0, "Quarterly": 60.0, "Yearly": 150.0}

# Coefficients of the "true" churn-risk model (on standardised inputs).
CHURN_EFFECTS: dict[str, float] = {
    "log_days_since_last_donation": 1.00,
    "email_open_rate": -0.45,
    "campaign_response_rate": -0.35,
    "recurring_donor": -0.80,
    "log_donor_tenure_months": -0.35,
    "previous_churn_flag": 0.60,
    "age": -0.15,
    "hidden_engagement": -0.30,
}
LIFE_EVENT_NOISE_SD: float = 0.75
MAX_DAYS_SINCE_LAST_DONATION: int = 1_095  # three years


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Logistic function that maps any real number to the range (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(values: np.ndarray) -> np.ndarray:
    """Scale values to mean 0 and standard deviation 1."""
    std = values.std()
    return (values - values.mean()) / std if std > 0 else values - values.mean()


def _calibrate_intercept(score: np.ndarray, target_rate: float) -> float:
    """Find the intercept so that the average churn probability equals ``target_rate``.

    Uses bisection, which is deterministic and needs no extra libraries.
    """
    low, high = -30.0, 30.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if np.mean(_sigmoid(mid + score)) > target_rate:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def _inject_missing(values: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
    """Return a copy of ``values`` with roughly ``rate`` of entries set to missing."""
    values = values.copy()
    if rate > 0:
        mask = rng.random(len(values)) < rate
        values[mask] = pd.NA if isinstance(values.dtype, pd.Int64Dtype) else np.nan
    return values


def generate_donor_dataset(
    n_donors: int = DEFAULT_N_DONORS,
    random_state: int = RANDOM_STATE,
    target_churn_rate: float = 0.25,
    missing_rate: float = 0.02,
) -> pd.DataFrame:
    """Create a synthetic donor dataset with realistic churn relationships.

    Args:
        n_donors: Number of donor records to create (must be >= 1).
        random_state: Seed for the random number generator (same seed -> same data).
        target_churn_rate: Approximate share of churned donors (between 0 and 1).
        missing_rate: Approximate share of missing values injected into
            ``age``, ``email_open_rate`` and ``campaign_response_rate``
            (``donation_frequency`` receives half this rate).

    Returns:
        A DataFrame with the columns listed in ``src.config.DATASET_COLUMNS``.
    """
    if n_donors < 1:
        raise ValueError(f"n_donors must be at least 1, got {n_donors}.")
    if not 0.0 < target_churn_rate < 1.0:
        raise ValueError(f"target_churn_rate must be between 0 and 1, got {target_churn_rate}.")
    if not 0.0 <= missing_rate < 0.5:
        raise ValueError(f"missing_rate must be in [0, 0.5), got {missing_rate}.")

    rng = np.random.default_rng(random_state)
    n = n_donors

    # 1. Demographics and giving pattern ------------------------------------
    age = np.clip(rng.normal(47, 14, n), 18, 90).round()
    frequency = rng.choice(DONATION_FREQUENCIES, size=n, p=FREQUENCY_PROBABILITIES)
    recurring = (rng.random(n) < np.vectorize(RECURRING_PROBABILITY.get)(frequency)).astype(int)

    # Older donors tend to have longer relationships with the organisation.
    age_tenure_factor = 0.6 + 0.8 * (age - 18) / 72
    tenure = np.clip(np.round(rng.gamma(2.0, 22.0, n) * age_tenure_factor) + 1, 1, 240).astype(int)

    # 2. Hidden engagement and observable engagement metrics ----------------
    engagement = rng.normal(0.0, 1.0, n) + 0.3 * recurring
    email_open_rate = 100 * _sigmoid(-0.4 + 0.9 * engagement + rng.normal(0, 0.6, n))
    campaign_response_rate = 100 * _sigmoid(-1.7 + 0.8 * engagement + rng.normal(0, 0.6, n))

    # 3. Donation history ----------------------------------------------------
    activity = np.clip(0.55 + 0.15 * engagement + 0.20 * recurring + rng.normal(0, 0.1, n), 0.1, 1.0)
    expected_gifts = tenure * np.vectorize(GIFTS_PER_MONTH.get)(frequency) * activity
    donation_count = np.maximum(1, rng.poisson(expected_gifts))

    gift_size = np.vectorize(TYPICAL_GIFT.get)(frequency) * rng.lognormal(0.0, 0.5, n)
    gift_size = np.maximum(5.0, gift_size * (1 + 0.006 * (age - 47)))
    total_donated = np.round(donation_count * gift_size, 2)
    avg_donation = np.round(total_donated / donation_count, 2)

    # Less engaged, non-recurring donors have longer gaps since their last gift.
    typical_gap = np.vectorize(TYPICAL_GAP_DAYS.get)(frequency)
    recency_factor = rng.lognormal(-0.3 - 0.35 * engagement - 0.30 * recurring, 0.7)
    max_possible_gap = np.minimum(MAX_DAYS_SINCE_LAST_DONATION, tenure * 30)
    days_since_last = np.clip(np.round(typical_gap * recency_factor), 1, max_possible_gap).astype(int)

    # Donors who lapsed in the past and were later won back.
    previous_churn = (
        rng.random(n) < _sigmoid(-2.2 + 0.012 * (tenure - 36) - 0.3 * engagement)
    ).astype(int)

    # 4. Churn label (what happens in the 12 months AFTER the snapshot) -----
    score = (
        CHURN_EFFECTS["log_days_since_last_donation"] * _standardize(np.log1p(days_since_last))
        + CHURN_EFFECTS["email_open_rate"] * _standardize(email_open_rate)
        + CHURN_EFFECTS["campaign_response_rate"] * _standardize(campaign_response_rate)
        + CHURN_EFFECTS["recurring_donor"] * recurring
        + CHURN_EFFECTS["log_donor_tenure_months"] * _standardize(np.log(tenure))
        + CHURN_EFFECTS["previous_churn_flag"] * previous_churn
        + CHURN_EFFECTS["age"] * _standardize(age)
        + CHURN_EFFECTS["hidden_engagement"] * engagement
        + rng.normal(0.0, LIFE_EVENT_NOISE_SD, n)
    )
    intercept = _calibrate_intercept(score, target_churn_rate)
    churn_probability = _sigmoid(intercept + score)
    churn = (rng.random(n) < churn_probability).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: [f"D{10_001 + i}" for i in range(n)],
            "age": pd.array(age.astype(int), dtype="Int64"),
            "donation_count": donation_count.astype(int),
            "total_donated": total_donated,
            "avg_donation": avg_donation,
            "days_since_last_donation": days_since_last,
            "donation_frequency": frequency,
            "donor_tenure_months": tenure,
            "email_open_rate": np.round(email_open_rate, 1),
            "campaign_response_rate": np.round(campaign_response_rate, 1),
            "recurring_donor": recurring,
            "previous_churn_flag": previous_churn,
            TARGET_COLUMN: churn,
        }
    )

    # 5. Missing values (random data-entry gaps, added after labelling) -----
    for column in ("age", "email_open_rate", "campaign_response_rate"):
        df[column] = _inject_missing(df[column], missing_rate, rng)
    df["donation_frequency"] = _inject_missing(
        df["donation_frequency"].astype(object), missing_rate / 2, rng
    )

    return df[DATASET_COLUMNS]


def save_dataset(df: pd.DataFrame, path: str | Path = RAW_DATA_PATH) -> Path:
    """Save the dataset as CSV, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def describe_dataset(df: pd.DataFrame) -> str:
    """Return a short human-readable summary of a generated dataset."""
    churned = int(df[TARGET_COLUMN].sum())
    total = len(df)
    lines = [
        f"Total donors:      {total:,}",
        f"Active donors:     {total - churned:,}",
        f"Churned donors:    {churned:,} ({churned / total:.1%})",
        f"Columns:           {df.shape[1]}",
        f"Missing values:    {int(df.isna().sum().sum()):,} cells",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic donor dataset.")
    parser.add_argument("--n-donors", type=int, default=DEFAULT_N_DONORS, help="Number of donors.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: generate and save the dataset."""
    args = _parse_args()
    df = generate_donor_dataset(n_donors=args.n_donors, random_state=args.seed)
    path = save_dataset(df, args.output)
    print("SYNTHETIC DONOR DATASET")
    print("=======================")
    print(describe_dataset(df))
    print(f"\nSaved to: {path}")
    print("Note: this is synthetic data for education only - not real donor records.")


if __name__ == "__main__":
    main()
