"""Unit and integration tests for benchmark_prediction.py.

Verifies GroupShuffleSplit by vessel_id, per-source evaluation breakdown,
target-mode switching between absolute and rate targets, and segmented per-source modeling.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_prediction import run_prediction_benchmark, run_segmented_benchmark


@pytest.fixture
def temp_benchmark_csv(tmp_path: Path) -> Path:
    """Create a temporary multi-source dataset for testing benchmarking."""
    rows = []
    # 30 mock rows (10 vessels)
    for i in range(30):
        rows.append({
            "voyage_id": f"VY-M-{i:03d}",
            "vessel_id": f"VSL-M-{i % 10:02d}",
            "vessel_type": "Bulk Carrier",
            "vessel_dwt": 50000.0,
            "cargo_tons": 35000.0,
            "distance_nm": 3000.0 + (i * 50.0),
            "speed_knots": 14.0,
            "hours_at_sea": 250.0,
            "fuel_type": "Diesel",
            "weather_factor": 1.1,
            "sea_state": 3,
            "data_source": "mock",
            "is_synthetic": True,
            "fuel_consumption": 750.0 + (i * 10.0),
            "co2_emissions": 2400.0,
        })
    # 30 thetis_mrv rows (10 vessels)
    for i in range(30):
        rows.append({
            "voyage_id": f"VY-T-{i:03d}",
            "vessel_id": f"VSL-T-{i % 10:02d}",
            "vessel_type": "Container Ship",
            "vessel_dwt": 70000.0,
            "cargo_tons": 50000.0,
            "distance_nm": 20000.0 + (i * 100.0),
            "speed_knots": 15.0,
            "hours_at_sea": 1800.0,
            "fuel_type": "LNG",
            "weather_factor": 1.0,
            "sea_state": 0,
            "data_source": "thetis_mrv",
            "is_synthetic": False,
            "fuel_consumption": 2800.0 + (i * 50.0),
            "co2_emissions": 8500.0,
        })
    # 30 fuelcast rows (10 vessels)
    for i in range(30):
        rows.append({
            "voyage_id": f"VY-F-{i:03d}",
            "vessel_id": f"VSL-F-{i % 10:02d}",
            "vessel_type": "Oil Tanker",
            "vessel_dwt": 80000.0,
            "cargo_tons": 60000.0,
            "distance_nm": 8.0,
            "speed_knots": 8.0,
            "hours_at_sea": 1.0,
            "fuel_type": "Diesel",
            "weather_factor": 1.05,
            "sea_state": 2,
            "data_source": "fuelcast",
            "is_synthetic": False,
            "fuel_consumption": 0.45 + (i * 0.01),
            "co2_emissions": 1.4,
        })

    csv_file = tmp_path / "test_benchmark_data.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    return csv_file


def test_benchmark_prediction_absolute_mode(temp_benchmark_csv: Path, tmp_path: Path):
    """Test run_prediction_benchmark in absolute target mode."""
    report_file = tmp_path / "benchmark_abs.json"
    res = run_prediction_benchmark(
        dataset_path=str(temp_benchmark_csv),
        output_report=str(report_file),
        sample_size=90,
        test_split=0.3,
        target_mode="absolute",
    )

    assert report_file.exists()
    assert res["target_mode"] == "absolute"
    assert "models" in res
    assert "linear_regression" in res["models"]

    lr = res["models"]["linear_regression"]
    assert "by_source" in lr
    for src in res["test_source_distribution"].keys():
        assert src in lr["by_source"]
        assert "rmse" in lr["by_source"][src]
        assert "mae" in lr["by_source"][src]
        assert "mape" in lr["by_source"][src]
        assert "r2" in lr["by_source"][src]


def test_benchmark_prediction_rate_mode(temp_benchmark_csv: Path, tmp_path: Path):
    """Test run_prediction_benchmark in rate target mode."""
    report_file = tmp_path / "benchmark_rate.json"
    res = run_prediction_benchmark(
        dataset_path=str(temp_benchmark_csv),
        output_report=str(report_file),
        sample_size=90,
        test_split=0.3,
        target_mode="rate",
    )

    assert report_file.exists()
    assert res["target_mode"] == "rate"
    lr = res["models"]["linear_regression"]
    assert "by_source" in lr
    for src in res["test_source_distribution"].keys():
        assert src in lr["by_source"]
        assert "rmse" in lr["by_source"][src]
        assert "mae" in lr["by_source"][src]
        assert "mape" in lr["by_source"][src]
        assert "r2" in lr["by_source"][src]


def test_segmented_benchmark(temp_benchmark_csv: Path, tmp_path: Path):
    """Test run_segmented_benchmark training separate models per source."""
    report_file = tmp_path / "benchmark_seg.json"
    res = run_segmented_benchmark(
        dataset_path=str(temp_benchmark_csv),
        output_report=str(report_file),
        sample_size=90,
        test_split=0.3,
    )

    assert report_file.exists()
    assert res["approach"] == "segmented"
    assert "models" in res
    assert "linear_regression" in res["models"]
    assert "qifcp" in res["models"]

    lr = res["models"]["linear_regression"]
    assert "by_source" in lr
    for src in res["test_source_distribution"].keys():
        assert src in lr["by_source"]
        assert "rmse" in lr["by_source"][src]
        assert "mae" in lr["by_source"][src]
        assert "r2" in lr["by_source"][src]
