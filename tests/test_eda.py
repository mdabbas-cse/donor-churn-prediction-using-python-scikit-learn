"""Tests for the EDA summary and chart generation."""

from __future__ import annotations

from src.config import TARGET_COLUMN
from src.eda import EDA_CHARTS, churn_rate_by_bin, compute_summary, format_summary, run_eda


def test_summary_counts_are_consistent(small_dataset):
    summary = compute_summary(small_dataset)
    assert summary["total_donors"] == len(small_dataset)
    assert summary["active_donors"] + summary["churned_donors"] == summary["total_donors"]
    assert summary["churn_rate"] == small_dataset[TARGET_COLUMN].mean()
    assert "Churn rate:" in format_summary(summary)


def test_churn_rate_by_bin(small_dataset):
    table = churn_rate_by_bin(small_dataset, "days_since_last_donation", [0, 90, 10_000], ["recent", "old"])
    assert list(table["band"]) == ["recent", "old"]
    assert table["donors"].sum() == len(small_dataset)
    assert table["churn_rate"].between(0, 1).all()


def test_run_eda_saves_all_charts(small_dataset, tmp_path):
    saved = run_eda(small_dataset, tmp_path, dpi=50)
    assert set(saved) == set(EDA_CHARTS)
    for path in saved.values():
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
