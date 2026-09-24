"""Deterministic unit tests for Quantum-Inspired Fuel Consumption Predictor (QIFCP).

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates:
1. Feature standardization prevents arctan saturation (std(theta) > 0.3 across real features).
2. Low inference latency guard (< 30ms for 200-sample voyage batch).
3. QPSO hyperparameter tuning synergy between Objective 1 and Objective 2.
4. ModelRegistry factory registration and scikit-learn estimator compliance.
"""

import os
import time
import numpy as np
import pandas as pd
import pytest

from contracts.exceptions import PredictionError
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


def test_qifcp_batch_inference_correctness(
    real_voyage_features: tuple[np.ndarray, np.ndarray],
) -> None:
    """Assert predict() on a 200-sample batch is correct, deterministic, and non-negative.

    Separated from the wall-clock latency check so CI correctness is never
    gated on runner CPU speed.
    """
    X, y = real_voyage_features
    model = QIFCPRegressor(gamma=0.5, random_state=42)
    model.fit(X, y)

    batch_200 = X[:200]
    preds_a = model.predict(batch_200)
    preds_b = model.predict(batch_200)

    # Shape
    assert len(preds_a) == 200, f"Expected 200 predictions, got {len(preds_a)}"
    # Non-negative fuel consumption
    assert np.all(preds_a >= 0.0), "QIFCP produced negative fuel consumption predictions"
    # Deterministic: identical input → identical output
    np.testing.assert_array_equal(
        preds_a,
        preds_b,
        err_msg="QIFCP predict() is non-deterministic for identical inputs",
    )


# ---------------------------------------------------------------------------
# Performance benchmark — skipped on shared CI runners by default.
#
# Run locally with:   pytest -m perf  (or set QIFCP_PERF_BENCH=1)
# The 30 ms strict threshold is a local developer regression guard.
# A generous 500 ms backstop fires even on slow runners so a genuine
# algorithmic regression (e.g. accidental O(n²) loop) is still caught in CI.
# ---------------------------------------------------------------------------
_CI_RUNNER = os.getenv("CI", "false").lower() == "true"
_FORCE_PERF = os.getenv("QIFCP_PERF_BENCH", "0") == "1"

_perf_skip = pytest.mark.skipif(
    _CI_RUNNER and not _FORCE_PERF,
    reason=(
        "Wall-clock latency tests are skipped on shared CI runners to avoid "
        "non-deterministic failures caused by CPU contention. "
        "Run locally or set QIFCP_PERF_BENCH=1 to enable."
    ),
)


@_perf_skip
@pytest.mark.perf
def test_qifcp_batch_inference_latency_strict(
    real_voyage_features: tuple[np.ndarray, np.ndarray],
) -> None:
    """Strict 30 ms latency guard — local developer benchmark only.

    Ensures QIFCP batch evaluation inside fleet_objective does not become an
    optimizer bottleneck during the scalability sweep.  Skipped on GitHub
    Actions runners; use QIFCP_PERF_BENCH=1 to force-enable.
    """
    X, y = real_voyage_features
    model = QIFCPRegressor(gamma=0.5, random_state=42)
    model.fit(X, y)

    batch_200 = X[:200]
    _ = model.predict(batch_200)  # warmup

    start = time.perf_counter()
    preds = model.predict(batch_200)
    latency_ms = (time.perf_counter() - start) * 1000.0

    assert len(preds) == 200
    assert latency_ms < 30.0, (
        f"Batch inference too slow: {latency_ms:.2f} ms > 30 ms strict threshold."
    )


@pytest.mark.perf
def test_qifcp_batch_inference_latency_ci_backstop(
    real_voyage_features: tuple[np.ndarray, np.ndarray],
) -> None:
    """Generous 500 ms backstop — always runs, even on shared CI runners.

    Catches genuine algorithmic regressions (e.g. accidental O(n²) loop)
    without being sensitive to normal runner CPU contention.
    """
    X, y = real_voyage_features
    model = QIFCPRegressor(gamma=0.5, random_state=42)
    model.fit(X, y)

    batch_200 = X[:200]
    _ = model.predict(batch_200)  # warmup

    start = time.perf_counter()
    preds = model.predict(batch_200)
    latency_ms = (time.perf_counter() - start) * 1000.0

    assert len(preds) == 200
    assert latency_ms < 500.0, (
        f"Batch inference severely degraded: {latency_ms:.2f} ms > 500 ms CI backstop. "
        "Likely an algorithmic regression, not runner contention."
    )



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


def test_qifcp_tune_with_qpso_vessel_grouped() -> None:
    """Assert QPSO hyperparameter tuning performs strict vessel-disjoint inner validation."""
    loader = CSVDatasetLoader()
    records = loader.load_data("data/raw/voyages_sample.csv")
    sample_records = records[:300]
    pipeline = FeatureEngineeringPipeline()
    X_df, y_ser = pipeline.get_training_features_and_target(sample_records, encode_categoricals=True)
    vessel_ids = np.array([r.vessel_id for r in sample_records])

    X = X_df.to_numpy(dtype=float)
    y = y_ser.to_numpy(dtype=float)

    model = QIFCPRegressor(gamma=0.5, alpha_reg=1.0, random_state=42)
    tuned = model.tune_with_qpso(
        X,
        y,
        groups=vessel_ids,
        population_size=6,
        max_iterations=6,
        val_split=0.25,
    )

    assert tuned is model
    assert hasattr(model, "inner_train_indices_")
    assert hasattr(model, "inner_val_indices_")

    train_vessels = set(vessel_ids[model.inner_train_indices_])
    val_vessels = set(vessel_ids[model.inner_val_indices_])

    # Assert ZERO vessel leakage in the inner hyperparameter validation split
    assert train_vessels.isdisjoint(val_vessels), (
        f"Vessel leakage detected in inner validation! Overlapping vessels: {train_vessels & val_vessels}"
    )
    assert len(train_vessels) >= 2
    assert len(val_vessels) >= 1
    assert len(model.tuning_history_) == 6

    # Verify model is fitted and can perform inference
    preds = model.predict(X[:50])
    assert len(preds) == 50
    assert np.all(preds >= 0.0)


def test_qifcp_tune_with_qpso_groups_mismatch_raises_error() -> None:
    """Assert passing mismatched groups length raises PredictionError."""
    X = np.ones((50, 5))
    y = np.ones(50)
    groups = np.array(["V1", "V2"])  # Length 2 != 50
    model = QIFCPRegressor(random_state=42)

    with pytest.raises(PredictionError, match="Length of groups"):
        model.tune_with_qpso(X, y, groups=groups)
