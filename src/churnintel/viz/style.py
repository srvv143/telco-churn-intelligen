"""Chart style tokens (validated categorical palette, light surface, hairline chrome)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DEEMPH = "#c3c2b7"  # de-emphasised series

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

STATUS = {"ok": "#0ca30c", "warn": "#fab219", "alert": "#d03b3b"}

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
# Diverging blue <-> red with a neutral gray midpoint ("reduces" vs "increases" churn risk).
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8f8e", "#c62f2f"]
)
POSITIVE = "#d03b3b"  # increases churn risk
NEGATIVE = BLUE  # reduces churn risk


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.labelcolor": INK2,
            "axes.labelsize": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK2,
            "lines.linewidth": 2.0,
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
