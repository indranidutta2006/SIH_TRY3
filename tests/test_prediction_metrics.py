"""Deterministic unit tests for prediction evaluation metrics.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A tests verifying mathematical accuracy of MAE, RMSE, MAPE, R2, zero-division safeguards,
model ranking logic, and dataclass serialization.
"""

import math
import numpy as np
import pytest

from contracts.exceptions import PredictionError
from src.prediction.metrics import (
    ModelComparisonResult,
    ModelMetrics,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_rmse,
    evaluate_predictions,
    select_best_model,
)


def test_compute_mae_correctness() -> None:
    """Verify MAE calculation on known test vectors."""
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 19.0, 33.0]
    # |10-12| = 2, |20-19| = 1, |30-33| = 3 -> (2 + 1 + 3) / 3 = 2.0
    mae = compute_mae(y_true, y_pred)
    assert pytest.approx(mae, rel=1e-5) == 2.0


def test_compute_rmse_correctness() -> None:
    """Verify RMSE calculation on known test vectors."""
    y_true = [10.0, 20.0]
    y_pred = [13.0, 24.0]
    # (3^2 + 4^2) / 2 = 25 / 2 = 12.5 -> sqrt(12.5) ~ 3.53553
    rmse = compute_rmse(y_true, y_pred)
    assert pytest.approx(rmse, rel=1e-4) == math.sqrt(12.5)


def test_compute_mape_correctness() -> None:
    """Verify MAPE calculation on known test vectors."""
    y_true = [100.0, 200.0]
    y_pred = [110.0, 180.0]
    # |(100-110)/100| = 10%, |(200-180)/200| = 10% -> MAPE = 10.0%
    mape = compute_mape(y_true, y_pred)
    assert pytest.approx(mape, rel=1e-4) == 10.0


def test_compute_mape_zero_division_safe() -> None:
    """Ensure MAPE handles zero values in actuals without crashing or returning NaN."""
    y_true = [0.0, 100.0]
    y_pred = [0.0, 100.0]
    mape = compute_mape(y_true, y_pred)
    assert not math.isnan(mape)
    assert not math.isinf(mape)


def test_compute_r2_correctness() -> None:
    """Verify R2 calculation for perfect predictions and baseline mean predictions."""
    y_true = [10.0, 20.0, 30.0, 40.0]

    # Perfect prediction -> R2 = 1.0
    r2_perfect = compute_r2(y_true, y_true)
    assert pytest.approx(r2_perfect, rel=1e-5) == 1.0

    # Mean prediction -> R2 = 0.0
    mean_val = float(np.mean(y_true))
    r2_mean = compute_r2(y_true, [mean_val] * len(y_true))
    assert pytest.approx(r2_mean, abs=1e-5) == 0.0


def test_shape_mismatch_raises_error() -> None:
    """Ensure differing lengths between true and predicted arrays raises PredictionError."""
    with pytest.raises(PredictionError) as exc_info:
        compute_rmse([1.0, 2.0], [1.0])
    assert "Shape mismatch" in str(exc_info.value)


def test_empty_dataset_raises_error() -> None:
    """Ensure empty arrays raise PredictionError."""
    with pytest.raises(PredictionError) as exc_info:
        compute_mae([], [])
    assert "empty dataset" in str(exc_info.value)


def test_evaluate_predictions_and_serialization() -> None:
    """Verify evaluate_predictions constructs a valid ModelMetrics dataclass."""
    y_true = [100.0, 200.0, 300.0]
    y_pred = [105.0, 195.0, 310.0]

    metrics = evaluate_predictions(y_true, y_pred, model_name="test_model")
    assert metrics.model_name == "test_model"
    assert metrics.mae > 0
    assert metrics.rmse > 0
    assert metrics.mape > 0
    assert metrics.r2 > 0

    # Test serialization
    data_dict = metrics.to_dict()
    assert data_dict["model_name"] == "test_model"
    assert "rmse" in data_dict
    assert isinstance(metrics.to_json(), str)


def test_select_best_model_empirical_selection() -> None:
    """Ensure select_best_model dynamically ranks models without hardcoding."""
    metrics_by_model = {
        "model_a": ModelMetrics("model_a", mae=15.0, rmse=20.0, mape=5.0, r2=0.85),
        "model_b": ModelMetrics("model_b", mae=8.0, rmse=10.0, mape=2.5, r2=0.96),
        "model_c": ModelMetrics("model_c", mae=12.0, rmse=16.0, mape=4.0, r2=0.90),
    }

    # Best by RMSE should be model_b (10.0 < 16.0 < 20.0)
    best_rmse = select_best_model(metrics_by_model, primary_metric="rmse")
    assert best_rmse == "model_b"

    # Best by R2 should be model_b (0.96 > 0.90 > 0.85)
    best_r2 = select_best_model(metrics_by_model, primary_metric="r2")
    assert best_r2 == "model_b"


def test_model_comparison_result_serialization() -> None:
    """Verify ModelComparisonResult serializes to valid JSON."""
    metrics = {
        "lr": ModelMetrics("lr", mae=10.0, rmse=14.0, mape=3.0, r2=0.92),
    }
    comp = ModelComparisonResult(
        best_model_name="lr",
        metrics_by_model=metrics,
        training_timestamp="2026-09-22T00:00:00Z",
    )
    json_str = comp.to_json()
    assert "best_model_name" in json_str
    assert "metrics_by_model" in json_str
