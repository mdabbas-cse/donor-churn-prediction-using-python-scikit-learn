"""Shared chart styling and figure-saving helpers.

Figures are created with :class:`matplotlib.figure.Figure` directly instead of
``pyplot``. This keeps chart creation headless (no GUI window is needed when
the scripts run in a terminal or in tests) while still displaying inline in a
Jupyter notebook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src.config import FIGURE_DPI

# Colour roles. Active/Churned use two hues that stay distinguishable for
# common forms of colour-vision deficiency; text always uses neutral ink.
COLORS: dict[str, str] = {
    "active": "#2a78d6",  # blue
    "churned": "#eb6834",  # orange
    "primary": "#2a78d6",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#ffffff",
}

STATUS_ORDER: list[str] = ["Active", "Churned"]
STATUS_PALETTE: dict[str, str] = {"Active": COLORS["active"], "Churned": COLORS["churned"]}

# Sequential (one hue, light -> dark) for magnitudes such as confusion-matrix counts.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "donor_blues", ["#f3f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)

# Diverging (blue <-> grey midpoint <-> red) for correlations from -1 to +1.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "donor_diverging", ["#104281", "#5598e7", "#f0efec", "#e66767", "#9e2a2a"]
)

_STYLE: dict[str, object] = {
    "figure.facecolor": COLORS["surface"],
    "axes.facecolor": COLORS["surface"],
    "savefig.facecolor": COLORS["surface"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "legend.title_fontsize": 10.5,
    "legend.frameon": False,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "text.color": COLORS["ink"],
    "axes.titlecolor": COLORS["ink"],
    "axes.labelcolor": COLORS["ink_secondary"],
    "axes.edgecolor": COLORS["axis"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "xtick.labelcolor": COLORS["ink_secondary"],
    "ytick.labelcolor": COLORS["ink_secondary"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.8,
    "savefig.dpi": FIGURE_DPI,
}


def apply_plot_style() -> None:
    """Apply the project's chart style to Matplotlib (safe to call repeatedly)."""
    mpl.rcParams.update(_STYLE)


def new_figure(
    figsize: tuple[float, float] = (8.0, 5.0), nrows: int = 1, ncols: int = 1, **subplot_kw
):
    """Create a styled figure and its axes.

    Returns:
        A ``(figure, axes)`` tuple. ``axes`` is a single Axes when
        ``nrows == ncols == 1``, otherwise a NumPy array of Axes.
    """
    apply_plot_style()
    fig = Figure(figsize=figsize, layout="constrained")
    axes = fig.subplots(nrows, ncols, **subplot_kw)
    return fig, axes


def save_figure(fig: Figure, output_path: str | Path, dpi: int = FIGURE_DPI) -> Path:
    """Save ``fig`` as a PNG, creating the parent folder if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=COLORS["surface"])
    return output_path


def percent_formatter(decimals: int = 0):
    """Return a tick formatter that shows 0-1 fractions as percentages."""
    return mpl.ticker.PercentFormatter(xmax=1.0, decimals=decimals)
