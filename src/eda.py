"""Exploratory Data Analysis (EDA) for the donor dataset.

Prints summary statistics and saves charts (300 DPI PNG) to ``reports/figures``.

Run from the project root::

    python -m src.eda
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.config import (
    BINARY_FEATURES,
    CLASS_LABELS,
    DONATION_FREQUENCIES,
    FIGURE_DPI,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.preprocess import load_dataset, validate_dataset
from src.visualization import (
    COLORS,
    DIVERGING_CMAP,
    STATUS_ORDER,
    STATUS_PALETTE,
    new_figure,
    percent_formatter,
    save_figure,
)

STATUS_COLUMN = "status"
LEGEND_TITLE = "Donor status"

RECENCY_BINS: list[float] = [0, 30, 90, 180, 365, 730, np.inf]
RECENCY_LABELS: list[str] = ["0-30", "31-90", "91-180", "181-365", "366-730", "730+"]
RATE_BINS: list[float] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
RATE_LABELS: list[str] = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-100"]
AGE_BINS: list[float] = [17, 29, 39, 49, 59, 69, 120]
AGE_LABELS: list[str] = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def compute_summary(df: pd.DataFrame) -> dict:
    """Compute the key descriptive statistics used in the EDA report."""
    total = len(df)
    churned = int((df[TARGET_COLUMN] == 1).sum())
    active = int((df[TARGET_COLUMN] == 0).sum())
    numeric_columns = NUMERIC_FEATURES + BINARY_FEATURES + [TARGET_COLUMN]
    return {
        "total_donors": total,
        "active_donors": active,
        "churned_donors": churned,
        "churn_rate": churned / total if total else float("nan"),
        "frequency_distribution": df["donation_frequency"].value_counts(dropna=False).to_dict(),
        "churn_rate_by_frequency": df.groupby("donation_frequency")[TARGET_COLUMN].mean().to_dict(),
        "churn_rate_by_recurring": df.groupby("recurring_donor")[TARGET_COLUMN].mean().to_dict(),
        "numeric_statistics": df[NUMERIC_FEATURES].describe().T,
        "means_by_status": df.groupby(TARGET_COLUMN)[NUMERIC_FEATURES].mean().T.rename(columns=CLASS_LABELS),
        "correlation_with_churn": (
            df[numeric_columns].corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()
        ),
    }


def format_summary(summary: dict) -> str:
    """Turn :func:`compute_summary` output into readable text."""
    lines = [
        "DONOR DATASET OVERVIEW",
        "======================",
        f"Total donors:    {summary['total_donors']:,}",
        f"Active donors:   {summary['active_donors']:,}",
        f"Churned donors:  {summary['churned_donors']:,}",
        f"Churn rate:      {summary['churn_rate']:.1%}",
        "",
        "Donation frequency (donors | churn rate):",
    ]
    for frequency, count in summary["frequency_distribution"].items():
        name = frequency if isinstance(frequency, str) else "Missing"
        rate = summary["churn_rate_by_frequency"].get(frequency)
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        lines.append(f"  {name:<10} {count:>6,} | {rate_text}")
    lines += ["", "Churn rate by recurring donor flag:"]
    for flag, rate in summary["churn_rate_by_recurring"].items():
        lines.append(f"  recurring_donor={flag}: {rate:.1%}")

    stats = summary["numeric_statistics"][["mean", "50%", "min", "max"]].rename(columns={"50%": "median"})
    lines += ["", "Numeric feature statistics:", stats.round(1).to_string()]
    lines += ["", "Average feature value by donor status:", summary["means_by_status"].round(1).to_string()]
    lines += [
        "",
        "Correlation with churn (Pearson, -1 to +1):",
        summary["correlation_with_churn"].round(3).to_string(),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _with_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable 'Active'/'Churned' column for chart legends."""
    out = df.copy()
    out[STATUS_COLUMN] = out[TARGET_COLUMN].map(CLASS_LABELS)
    return out


def churn_rate_by_bin(
    df: pd.DataFrame, column: str, bins: Sequence[float], labels: Sequence[str]
) -> pd.DataFrame:
    """Churn rate and donor count for each value band of ``column``."""
    bands = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    grouped = df.groupby(bands, observed=True)[TARGET_COLUMN].agg(["mean", "size"])
    return grouped.rename(columns={"mean": "churn_rate", "size": "donors"}).reset_index(names="band")


def _plot_rate_bars(ax: Axes, rates: pd.DataFrame, overall_rate: float, xlabel: str, title: str) -> None:
    """Bar chart of churn rate per band with an overall-average reference line."""
    x = np.arange(len(rates))
    bars = ax.bar(x, rates["churn_rate"], color=COLORS["churned"], width=0.7, edgecolor="white", linewidth=1.5)
    ax.bar_label(
        bars,
        labels=[f"{v:.0%}" for v in rates["churn_rate"]],
        padding=3,
        fontsize=9.5,
        color=COLORS["ink_secondary"],
    )
    ax.axhline(overall_rate, color=COLORS["ink_secondary"], linestyle="--", linewidth=1.2,
               label=f"Overall churn rate ({overall_rate:.1%})")
    ax.set_xticks(x, rates["band"].astype(str))
    ax.set_ylim(0, min(1.0, max(rates["churn_rate"].max(), overall_rate) * 1.2 + 0.02))
    ax.yaxis.set_major_formatter(percent_formatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Churn rate (% of donors who churned)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")


def _status_hist(ax: Axes, data: pd.DataFrame, column: str, **kwargs) -> None:
    """Overlaid density histograms for active vs churned donors."""
    sns.histplot(
        data=data,
        x=column,
        hue=STATUS_COLUMN,
        hue_order=STATUS_ORDER,
        palette=STATUS_PALETTE,
        stat="density",
        common_norm=False,
        element="step",
        alpha=0.25,
        linewidth=1.5,
        ax=ax,
        **kwargs,
    )
    sns.move_legend(ax, "upper right", title=LEGEND_TITLE)
    ax.set_ylabel("Density (share of donors in each group)")


# --------------------------------------------------------------------------- #
# Charts (each returns a Matplotlib Figure)
# --------------------------------------------------------------------------- #
def plot_churn_distribution(df: pd.DataFrame) -> Figure:
    """1. Share of active vs churned donors as a 100% stacked bar."""
    counts = df[TARGET_COLUMN].map(CLASS_LABELS).value_counts().reindex(STATUS_ORDER, fill_value=0)
    shares = counts / counts.sum()
    fig, ax = new_figure(figsize=(10, 3.2))
    left = 0.0
    for status in STATUS_ORDER:
        ax.barh([0], [shares[status]], left=left, color=STATUS_PALETTE[status], height=0.55,
                edgecolor="white", linewidth=2, label=status)
        ax.text(left + shares[status] / 2, 0, f"{status}\n{shares[status]:.1%}  ({counts[status]:,} donors)",
                ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
        left += shares[status]
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.set_xlabel("Share of all donors")
    ax.set_title(f"Donor churn distribution ({counts.sum():,} donors)")
    ax.grid(axis="y", visible=False)
    ax.legend(title=LEGEND_TITLE, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2)
    return fig


def plot_active_vs_churned(df: pd.DataFrame) -> Figure:
    """2. Bar chart comparing the number of active and churned donors."""
    counts = df[TARGET_COLUMN].map(CLASS_LABELS).value_counts().reindex(STATUS_ORDER, fill_value=0)
    fig, ax = new_figure(figsize=(7, 5.5))
    bars = ax.bar(STATUS_ORDER, counts.values, color=[STATUS_PALETTE[s] for s in STATUS_ORDER],
                  width=0.55)
    ax.bar_label(bars, labels=[f"{c:,}\n({c / counts.sum():.1%})" for c in counts.values],
                 padding=4, fontsize=11, color=COLORS["ink"])
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_xlabel("Donor status (0 = Active, 1 = Churned)")
    ax.set_ylabel("Number of donors")
    ax.set_title("Active vs churned donors")
    ax.grid(axis="x", visible=False)
    handles = [bar for bar in bars]
    ax.legend(handles, STATUS_ORDER, title=LEGEND_TITLE, loc="upper right")
    return fig


def plot_donation_frequency(df: pd.DataFrame) -> Figure:
    """3. Donor counts and churn rate by donation frequency."""
    data = _with_status(df.dropna(subset=["donation_frequency"]))
    fig, (ax_count, ax_rate) = new_figure(figsize=(13, 5.2), ncols=2)

    sns.countplot(data=data, x="donation_frequency", hue=STATUS_COLUMN, order=list(DONATION_FREQUENCIES),
                  hue_order=STATUS_ORDER, palette=STATUS_PALETTE, saturation=1, edgecolor="white", linewidth=1.5,
                  ax=ax_count)
    for container in ax_count.containers:
        ax_count.bar_label(container, fmt="{:,.0f}", padding=3, fontsize=9.5, color=COLORS["ink_secondary"])
    sns.move_legend(ax_count, "upper right", title=LEGEND_TITLE)
    ax_count.set_xlabel("Donation frequency")
    ax_count.set_ylabel("Number of donors")
    ax_count.set_title("Donors by donation frequency")
    ax_count.grid(axis="x", visible=False)

    rates = (
        data.groupby("donation_frequency")[TARGET_COLUMN].agg(churn_rate="mean", donors="size")
        .reindex(list(DONATION_FREQUENCIES)).reset_index(names="band")
    )
    _plot_rate_bars(ax_rate, rates, df[TARGET_COLUMN].mean(), "Donation frequency",
                    "Churn rate by donation frequency")
    fig.suptitle("Donation frequency distribution", x=0.01, ha="left")
    return fig


def plot_recency(df: pd.DataFrame) -> Figure:
    """4. Days since last donation: distribution and churn rate by recency band."""
    data = _with_status(df)
    fig, (ax_hist, ax_rate) = new_figure(figsize=(13, 5.2), ncols=2)
    _status_hist(ax_hist, data, "days_since_last_donation", binwidth=30)
    ax_hist.set_xlabel("Days since last donation")
    ax_hist.set_title("Distribution of days since last donation")

    rates = churn_rate_by_bin(df, "days_since_last_donation", RECENCY_BINS, RECENCY_LABELS)
    _plot_rate_bars(ax_rate, rates, df[TARGET_COLUMN].mean(), "Days since last donation",
                    "Churn rate rises as donors go longer without giving")
    fig.suptitle("Churn by donation recency", x=0.01, ha="left")
    return fig


def plot_donation_amounts(df: pd.DataFrame) -> Figure:
    """5. Distribution of average and lifetime donation amounts (log scale)."""
    data = _with_status(df)
    fig, (ax_avg, ax_total) = new_figure(figsize=(13, 5.2), ncols=2)
    _status_hist(ax_avg, data, "avg_donation", log_scale=True, bins=40)
    ax_avg.set_xlabel("Average donation ($, log scale)")
    ax_avg.set_title("Average donation amount")
    _status_hist(ax_total, data, "total_donated", log_scale=True, bins=40)
    ax_total.set_xlabel("Total donated over lifetime ($, log scale)")
    ax_total.set_title("Lifetime donation amount")
    fig.suptitle("Donation amount distribution", x=0.01, ha="left")
    return fig


def _plot_engagement(df: pd.DataFrame, column: str, label: str, title: str) -> Figure:
    """Box plot by status plus churn rate by engagement band."""
    data = _with_status(df.dropna(subset=[column]))
    fig, (ax_box, ax_rate) = new_figure(figsize=(13, 5.2), ncols=2)
    sns.boxplot(data=data, x=STATUS_COLUMN, y=column, hue=STATUS_COLUMN, order=STATUS_ORDER,
                hue_order=STATUS_ORDER, palette=STATUS_PALETTE, saturation=1, width=0.5, fliersize=2, legend=False,
                ax=ax_box)
    medians = data.groupby(STATUS_COLUMN)[column].median()
    for i, status in enumerate(STATUS_ORDER):
        ax_box.annotate(f"median {medians[status]:.1f}%", xy=(i + 0.27, medians[status]),
                        va="center", fontsize=9.5, color=COLORS["ink_secondary"])
    ax_box.set_xlabel(LEGEND_TITLE)
    ax_box.set_ylabel(f"{label} (%)")
    ax_box.set_title(f"{label} by donor status")
    ax_box.grid(axis="x", visible=False)
    ax_box.set_ylim(0, 118)
    ax_box.legend(handles=[ax_box.patches[i] for i in range(len(STATUS_ORDER))], labels=STATUS_ORDER,
                  title=LEGEND_TITLE, loc="upper center", ncol=2)

    rates = churn_rate_by_bin(df, column, RATE_BINS, RATE_LABELS)
    _plot_rate_bars(ax_rate, rates, df[TARGET_COLUMN].mean(), f"{label} band (%)", f"Churn rate by {label.lower()}")
    fig.suptitle(title, x=0.01, ha="left")
    return fig


def plot_email_engagement(df: pd.DataFrame) -> Figure:
    """6. Email open rate vs churn."""
    return _plot_engagement(df, "email_open_rate", "Email open rate", "Email engagement vs churn")


def plot_campaign_response(df: pd.DataFrame) -> Figure:
    """Extra: campaign response rate vs churn."""
    return _plot_engagement(df, "campaign_response_rate", "Campaign response rate", "Campaign response vs churn")


def plot_age_distribution(df: pd.DataFrame) -> Figure:
    """Extra: donor age distribution and churn rate by age group."""
    data = _with_status(df)
    fig, (ax_hist, ax_rate) = new_figure(figsize=(13, 5.2), ncols=2)
    _status_hist(ax_hist, data, "age", binwidth=5)
    ax_hist.set_xlabel("Donor age (years)")
    ax_hist.set_title("Age distribution by donor status")
    rates = churn_rate_by_bin(df, "age", AGE_BINS, AGE_LABELS)
    _plot_rate_bars(ax_rate, rates, df[TARGET_COLUMN].mean(), "Age group (years)", "Churn rate by age group")
    fig.suptitle("Donor age distribution", x=0.01, ha="left")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """7. Pearson correlation between numerical variables (lower triangle)."""
    columns = NUMERIC_FEATURES + BINARY_FEATURES + [TARGET_COLUMN]
    corr = df[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = new_figure(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"fontsize": 9}, linewidths=1.5, linecolor="white", square=True,
                cbar_kws={"label": "Correlation coefficient (-1 to +1)", "shrink": 0.75}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation between numerical variables")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Variable")
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    return fig


# File name -> chart function. Order follows the README gallery.
EDA_CHARTS: dict[str, Callable[[pd.DataFrame], Figure]] = {
    "donor_churn_distribution.png": plot_churn_distribution,
    "active_vs_churned_donors.png": plot_active_vs_churned,
    "donation_frequency.png": plot_donation_frequency,
    "churn_by_donation_recency.png": plot_recency,
    "donation_amount_distribution.png": plot_donation_amounts,
    "email_engagement_vs_churn.png": plot_email_engagement,
    "campaign_response_vs_churn.png": plot_campaign_response,
    "donor_age_distribution.png": plot_age_distribution,
    "correlation_heatmap.png": plot_correlation_heatmap,
}


def run_eda(df: pd.DataFrame, figures_dir: str | Path = FIGURES_DIR, dpi: int = FIGURE_DPI) -> dict[str, Path]:
    """Create and save every EDA chart. Returns ``{file name: saved path}``."""
    figures_dir = Path(figures_dir)
    saved: dict[str, Path] = {}
    for file_name, plot_function in EDA_CHARTS.items():
        fig = plot_function(df)
        saved[file_name] = save_figure(fig, figures_dir / file_name, dpi=dpi)
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Dataset CSV path.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Folder for PNG charts.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    df = load_dataset(args.data)
    print("DATA QUALITY")
    print("============")
    print(validate_dataset(df).to_text())
    print()
    print(format_summary(compute_summary(df)))
    saved = run_eda(df, args.figures_dir)
    print(f"\nSaved {len(saved)} charts to {args.figures_dir}:")
    for path in saved.values():
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
