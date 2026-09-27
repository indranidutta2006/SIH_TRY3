"""Unit tests verifying dashboard prediction visualization consistency with canonical benchmark artifacts.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Ensures that the dashboard data loader and visualization pipeline faithfully transform
outputs/reports/final_prediction_benchmark.json into the expected models, metrics, and status labels.
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from src.prediction.benchmark_loader import (
    DEFAULT_CANONICAL_BENCHMARK_PATH,
    get_canonical_benchmark_metadata,
    get_canonical_prediction_dataframe,
    load_canonical_prediction_benchmark,
)


def test_canonical_prediction_benchmark_file_exists() -> None:
    """Verify that canonical final prediction benchmark JSON report exists on disk."""
    assert DEFAULT_CANONICAL_BENCHMARK_PATH.exists(), (
        f"Missing canonical benchmark file: {DEFAULT_CANONICAL_BENCHMARK_PATH}"
    )


def test_canonical_benchmark_metadata_extraction() -> None:
    """Verify metadata extraction adheres to reporting and context guidelines."""
    data = load_canonical_prediction_benchmark()
    meta = get_canonical_benchmark_metadata(data)

    assert meta.dataset_path == "data/raw/voyages_sample.csv"
    assert meta.seeds == [42, 43, 44, 45, 46]
    assert "5-seed vessel-disjoint evaluation" in meta.subtitle
    assert "GroupShuffleSplit by vessel_id" in meta.subtitle
    assert meta.accuracy_claim == "Physics + Residual QIFCP — canonical 5-seed result"


def test_dashboard_dataframe_models_and_status() -> None:
    """Verify exact 5 canonical models and status labels are preserved in order."""
    df = get_canonical_prediction_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 5

    model_names = list(df["Model Architecture"])
    expected_models = [
        "Physics + Residual QIFCP",
        "Naval Physics Baseline",
        "Random Forest",
        "HistGradientBoosting",
        "Grouped QIFCP Direct",
    ]
    for expected in expected_models:
        assert expected in model_names, f"Expected model '{expected}' not found in dataframe"

    # Verify no ambiguous 'QIFCP' label
    for name in model_names:
        assert name != "QIFCP", "Ambiguous model name 'QIFCP' must not be displayed"

    # Verify status labels
    status_map = dict(zip(df["Model Architecture"], df["Status"]))
    assert status_map["Physics + Residual QIFCP"] == "PRODUCTION / FROZEN"
    assert status_map["Naval Physics Baseline"] == "REFERENCE"
    assert status_map["Random Forest"] == "REFERENCE"
    assert status_map["HistGradientBoosting"] == "REFERENCE"
    assert status_map["Grouped QIFCP Direct"] == "ABLATION"


def test_dashboard_metric_values_match_canonical_json() -> None:
    """Verify metric values in dataframe match aggregate_results from final_prediction_benchmark.json."""
    raw_data = load_canonical_prediction_benchmark()
    agg = raw_data["aggregate_results"]
    df = get_canonical_prediction_dataframe(raw_data)

    # Convert dataframe to dictionary by Model Key
    df_by_key = {row["Model Key"]: row for _, row in df.iterrows()}

    # Expected canonical values
    expected_specs = {
        "physics_residual_qifcp": {
            "name": "Physics + Residual QIFCP",
            "r2": 0.9933,
            "rmse": 61.55,
            "mae": 40.07,
        },
        "physics_baseline": {
            "name": "Naval Physics Baseline",
            "r2": 0.9897,
            "rmse": 79.00,
            "mae": 50.79,
        },
        "random_forest": {
            "name": "Random Forest",
            "r2": 0.9625,
            "rmse": 149.03,
            "mae": 99.90,
        },
        "hist_gradient_boosting": {
            "name": "HistGradientBoosting",
            "r2": 0.9529,
            "rmse": 161.00,
            "mae": 99.53,
        },
        "adaptive_grouped_gamma": {
            "name": "Grouped QIFCP Direct",
            "r2": 0.9567,
            "rmse": 156.40,
            "mae": 110.88,
        },
    }

    for key, exp in expected_specs.items():
        assert key in df_by_key, f"Model key '{key}' missing from dataframe"
        row = df_by_key[key]

        # Verify matched display name
        assert row["Model Architecture"] == exp["name"]

        # Verify values match JSON raw mean rounded
        raw_r2 = round(float(agg[key]["r2"]["mean"]), 4)
        raw_rmse = round(float(agg[key]["rmse"]["mean"]), 2)
        raw_mae = round(float(agg[key]["mae"]["mean"]), 2)

        assert row["R² Score"] == raw_r2 == exp["r2"]
        assert row["RMSE (tons)"] == raw_rmse == exp["rmse"]
        assert row["MAE (tons)"] == raw_mae == exp["mae"]


def test_dashboard_page_prediction_imports_cleanly() -> None:
    """Verify that page_prediction module imports cleanly and renders without syntax/runtime errors."""
    from app.dashboard.page_prediction import render_prediction_page
    assert callable(render_prediction_page)
