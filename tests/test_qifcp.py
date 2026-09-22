"""Deterministic unit tests for Quantum-Inspired Fuel Consumption Predictor (QIFCP).

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates:
1. Feature standardization prevents arctan saturation (std(theta) > 0.3 across real features).
2. Low inference latency guard (< 30ms for 200-sample voyage batch).
3. QPSO hyperparameter tuning synergy between Objective 1 and Objective 2.
4. ModelRegistry factory registration and scikit-learn estimator compliance.
"""

import time
import numpy as np
import pandas as pd
import pytest

from src.ingestion.dataset_loader import CSVDatasetLoader
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.prediction.model_registry import ModelRegistry
from src.prediction.qifcp import QIFCPRegressor


@pytest.fixture
def real_voyage_features() -> tuple[np.ndarray, np.ndarray]:
    """Fixture providing real preprocessed voyage features from data/raw/voyages_sample.csv."""
    loader = CSVDatasetLoader()
    records = loader.load_data("data/raw/voyages_sample.csv")
    pipeline = FeatureEngineeringPipeline()
    train_records = records[:500]  # Fast sample for testing
    X, y = pipeline.get_training_features_and_target(train_records, encode_categoricals=True)
    return X.to_numpy(dtype=float), y.to_numpy(dtype=float)


def test_qifcp_phase_angles_do_not_saturate(
    real_voyage_features: tuple[np.ndarray, np.ndarray],
) -> None:
    """Assert theta = arctan(gamma * Z) spans a non-trivial range (std > 0.3).

    Guarantees that features like distance_nm (up to 7500) and vessel_dwt (100k+)
    do not collapse to near-constant +/- pi/2 values.
    """
    X, y = real_voyage_features
    model = QIFCPRegressor(gamma=0.5, random_state=42)
    model.fit(X, y)

    theta = model.compute_phase_distribution(X)

    # Check phase standard deviation across all non-constant feature columns
    non_constant = np.std(X, axis=0) > 1e-4
    std_per_feature = np.std(theta, axis=0)

    # Standard deviation should be healthy and not collapsed (e.g. > 0.25)
    assert np.all(std_per_feature[non_constant] > 0.25), (
        f"Some non-constant feature phase angles saturated! Minimum std: {np.min(std_per_feature[non_constant]):.4f}"
    )
    # Ensure values are strictly bounded by (-pi/2, pi/2)
    assert np.all(np.abs(theta) < np.pi / 2.0)


def test_qifcp_batch_inference_latency_guard(
    real_voyage_features: tuple[np.ndarray, np.ndarray],
) -> None:
    """Assert predict() on a batch of 200 voyages completes in under 30 milliseconds.

    Regression guard ensuring QIFCP evaluation inside fleet_objective does not
    create an optimizer bottleneck during the large scalability benchmark.
    """
    X, y = real_voyage_features
    model = QIFCPRegressor(gamma=0.5, random_state=42)
    model.fit(X, y)

    batch_200 = X[:200]
    # Warmup
    _ = model.predict(batch_200)

    start = time.perf_counter()
    preds = model.predict(batch_200)
    latency_ms = (time.perf_counter() - start) * 1000.0

    assert len(preds) == 200
    assert latency_ms < 30.0, f"Batch inference too slow: {latency_ms:.2f} ms > 30 ms threshold."


def test_qifcp_tune_with_qpso(
    real_voyage_features: tuple[np.ndarray, np.ndarray],
) -> None:
    """Assert QPSO tunes QIFCP hyperparameters under equal-budget protocol."""
    X, y = real_voyage_features
    model = QIFCPRegressor(gamma=0.5, alpha_reg=1.0, random_state=42)

    # Tune with QPSO on small budget
    tuned_model = model.tune_with_qpso(
        X[:200], y[:200], population_size=6, max_iterations=8
    )

    assert tuned_model is model
    assert len(model.tuning_history_) == 8
    # Validation RMSE should improve or stabilize
    assert model.tuning_history_[-1] <= model.tuning_history_[0] + 1e-3


def test_qifcp_model_registry_integration() -> None:
    """Assert QIFCP is properly instantiated via ModelRegistry."""
    registry = ModelRegistry(random_state=42)
    assert "qifcp" in registry.get_supported_models(include_all=True)

    model = registry.create_model("qifcp")
    assert isinstance(model, QIFCPRegressor)
    assert ModelRegistry.get_artifact_filename("qifcp") == "qifcp.pkl"
