"""Sensitivity matrix and heatmap output helpers."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_sensitivity_matrix(
    metric_series: pd.Series,
    *,
    fast_windows: list[int],
    slow_windows: list[int],
) -> pd.DataFrame:
    """Project a (fast, slow) metric series to a tabular matrix."""
    matrix = pd.DataFrame(index=fast_windows, columns=slow_windows, dtype=float)
    for (fast, slow), value in metric_series.items():
        if fast in matrix.index and slow in matrix.columns:
            matrix.loc[fast, slow] = float(value)
    return matrix


def save_sensitivity_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    *,
    metric_name: str,
) -> None:
    """Render and save a PNG heatmap for parameter sensitivity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Slow MA window")
    ax.set_ylabel("Fast MA window")
    ax.set_title(f"Sensitivity Heatmap ({metric_name})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
