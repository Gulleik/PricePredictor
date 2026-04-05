"""Scenario tests for sensitivity matrix and heatmap generation."""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.sensitivity import build_sensitivity_matrix, save_sensitivity_heatmap

pytestmark = pytest.mark.scenario


def test_build_sensitivity_matrix_shape_and_cells() -> None:
    """Matrix should map metric values to (fast, slow) coordinates."""
    metric = pd.Series(
        [1.1, 0.6, 0.2],
        index=pd.MultiIndex.from_tuples([(5, 30), (10, 40), (15, 60)]),
    )

    matrix = build_sensitivity_matrix(
        metric,
        fast_windows=[5, 10, 15],
        slow_windows=[30, 40, 60],
    )

    assert matrix.shape == (3, 3)
    assert matrix.loc[5, 30] == 1.1
    assert matrix.loc[10, 40] == 0.6
    assert matrix.loc[15, 60] == 0.2
    assert pd.isna(matrix.loc[5, 60])


def test_save_sensitivity_heatmap_creates_png(tmp_path: Path) -> None:
    """Heatmap writer should create a non-empty PNG artifact."""
    matrix = pd.DataFrame(
        [[1.0, 0.5], [0.7, 0.2]],
        index=[5, 10],
        columns=[30, 40],
    )
    output_path = tmp_path / "heatmap.png"

    save_sensitivity_heatmap(
        matrix,
        output_path,
        metric_name="sharpe_ratio",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
