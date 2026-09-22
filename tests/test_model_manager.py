"""Deterministic unit tests for ProductionModelManager.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2B tests verifying artifact discovery, metrics manifest parsing, dynamic best model deployment,
in-memory engine caching, and error handling for missing models.
"""

from pathlib import Path
import pytest

from contracts.exceptions import PredictionError
from contracts.schemas import PredictionResult, VoyageRecord
from src.prediction.metrics import ModelMetrics
from src.prediction.model_manager import ProductionModelManager
from src.prediction.predictor import PredictionInferenceEngine


def test_list_available_models() -> None:
    """Ensure manager discovers existing persisted baseline model artifacts."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    available = manager.list_available_models()

    assert "hist_gradient_boosting" in available
    assert "linear_regression" in available
    assert "random_forest" in available


def test_get_all_model_metrics() -> None:
    """Ensure manager parses all metrics from baseline_metrics.json."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    all_metrics = manager.get_model_metrics()

    assert isinstance(all_metrics, dict)
    assert len(all_metrics) == 3
    assert "hist_gradient_boosting" in all_metrics
    assert isinstance(all_metrics["hist_gradient_boosting"], ModelMetrics)
    assert all_metrics["hist_gradient_boosting"].r2 > 0.95


def test_get_specific_model_metrics() -> None:
    """Ensure manager retrieves metrics for a single specified model identifier."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    hgbt_metrics = manager.get_model_metrics("HistGradientBoostingRegressor")

    assert isinstance(hgbt_metrics, ModelMetrics)
    assert hgbt_metrics.model_name == "hist_gradient_boosting"
    assert hgbt_metrics.rmse > 0


def test_get_best_model_name() -> None:
    """Ensure manager resolves best model name from manifest."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    best_name = manager.get_best_model_name()
    assert best_name == "hist_gradient_boosting"


def test_load_best_model() -> None:
    """Ensure get_best_model auto-deploys the best performing regressor."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    best_engine = manager.get_best_model()

    assert isinstance(best_engine, PredictionInferenceEngine)
    assert best_engine.model_name == "hist_gradient_boosting"


def test_load_model_by_name() -> None:
    """Ensure loading individual model by canonical or alias name works."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    lr_engine = manager.load_model("linear_regression")

    assert isinstance(lr_engine, PredictionInferenceEngine)
    assert lr_engine.model_name == "linear_regression"


def test_model_caching() -> None:
    """Ensure repeated load_model requests return the identical cached instance."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    engine1 = manager.load_model("random_forest")
    engine2 = manager.load_model("random_forest")

    assert engine1 is engine2


def test_missing_metrics_manifest_raises_error(tmp_path: Path) -> None:
    """Ensure manager raises PredictionError when metrics manifest is absent."""
    manager = ProductionModelManager(artifacts_dir=tmp_path)
    with pytest.raises(PredictionError) as exc_info:
        manager.get_model_metrics()

    assert "does not exist" in str(exc_info.value)


def test_nonexistent_model_raises_error() -> None:
    """Ensure requesting unknown model binary raises PredictionError."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    with pytest.raises(PredictionError) as exc_info:
        manager.load_model("nonexistent_deep_net")

    assert "Unknown model name" in str(exc_info.value) or "is not registered" in str(exc_info.value)


def test_inference_using_best_model() -> None:
    """Verify executing predictions with the auto-loaded production model."""
    manager = ProductionModelManager(artifacts_dir="artifacts")
    engine = manager.get_best_model()

    unlabelled_voyage = VoyageRecord(
        voyage_id="VY-PROD-001",
        vessel_id="VSL-001",
        vessel_type="Bulk Carrier",
        vessel_dwt=120000.0,
        cargo_tons=95000.0,
        distance_nm=4000.0,
        speed_knots=13.2,
        hours_at_sea=303.03,
        fuel_type="Diesel",
        weather_factor=1.12,
        sea_state=3,
        data_source="mock",
        is_synthetic=True,
        fuel_consumption=None,
        co2_emissions=None,
    )

    results = engine.predict([unlabelled_voyage])
    assert len(results) == 1
    assert isinstance(results[0], PredictionResult)
    assert results[0].predicted_fuel_consumption > 0.0
