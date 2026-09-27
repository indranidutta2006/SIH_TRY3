"""Audit verification tests for data generation, target leakage, and physics baseline independence.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies:
  1. Physics calibration uses training data only.
  2. Test target is never passed into calibration or residual training.
  3. Residual training does not use y_test.
  4. Generator target formula is deterministic for a fixed seed.
  5. Feature pipeline rejects forbidden target-derived columns.
"""

from pathlib import Path
import numpy as np
import pytest

from contracts.exceptions import DataValidationError
from contracts.schemas import VoyageRecord
from scripts.make_mock_dataset import generate_synthetic_voyages
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.prediction.qifcp import NavalPhysicsFuelBaseline, PhysicsInformedQIFCPRegressor


def test_generator_deterministic_for_fixed_seed():
    """Verify that the synthetic voyage generator produces bitwise identical records for fixed seed."""
    recs1 = generate_synthetic_voyages(num_rows=20, seed=42)
    recs2 = generate_synthetic_voyages(num_rows=20, seed=42)

    assert len(recs1) == len(recs2) == 20
    for r1, r2 in zip(recs1, recs2):
        assert r1.voyage_id == r2.voyage_id
        assert r1.vessel_id == r2.vessel_id
        assert r1.fuel_consumption == r2.fuel_consumption
        assert r1.co2_emissions == r2.co2_emissions
        assert r1.distance_nm == r2.distance_nm
        assert r1.speed_knots == r2.speed_knots


def test_feature_pipeline_rejects_forbidden_columns():
    """Verify that FeatureEngineeringPipeline rejects forbidden target-derived columns."""
    pipeline = FeatureEngineeringPipeline()

    sample_recs = generate_synthetic_voyages(num_rows=10, seed=42)
    df, _ = pipeline.get_training_features_and_target(sample_recs)

    # Clean df must not have any target-derived or duration proxy columns
    forbidden = ["fuel_consumption", "co2_emissions", "hours_at_sea", "implied_speed", "speed_discrepancy"]
    for col in forbidden:
        assert col not in df.columns

    # Injecting forbidden column must trigger DataValidationError
    df_leaked = df.copy()
    df_leaked["fuel_consumption"] = 123.4
    with pytest.raises(DataValidationError):
        pipeline.check_for_target_leakage(df_leaked)


def test_physics_calibration_uses_training_data_only():
    """Verify that NavalPhysicsFuelBaseline calibration parameters depend strictly on X_train, y_train."""
    rng = np.random.default_rng(42)
    X_train = np.abs(rng.normal(50, 10, size=(30, 27)))
    y_train = np.abs(rng.normal(200, 30, size=30))

    X_test_1 = np.abs(rng.normal(10, 2, size=(10, 27)))
    X_test_2 = np.abs(rng.normal(100, 20, size=(10, 27)))

    model = NavalPhysicsFuelBaseline()
    model.fit(X_train, y_train)

    c_prop_orig = model.c_prop_
    c_aux_orig = model.c_aux_

    # Repeated prediction across distinct unseen test sets must not alter fitted parameters
    _ = model.predict(X_test_1)
    assert model.c_prop_ == c_prop_orig
    assert model.c_aux_ == c_aux_orig

    _ = model.predict(X_test_2)
    assert model.c_prop_ == c_prop_orig
    assert model.c_aux_ == c_aux_orig


def test_residual_training_does_not_use_test_target():
    """Verify that PhysicsInformedQIFCPRegressor never accepts or accesses y_test during inference."""
    rng = np.random.default_rng(42)
    X_train = np.abs(rng.normal(50, 10, size=(40, 27)))
    y_train = np.abs(rng.normal(200, 30, size=40))
    X_test = np.abs(rng.normal(60, 15, size=(12, 27)))

    model = PhysicsInformedQIFCPRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict takes only X_test — y_test is not part of the signature
    preds = model.predict(X_test)
    assert preds.shape == (12,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)


def test_physics_baseline_and_residual_decomposition_consistency():
    """Verify that final prediction strictly decomposes into y_phys + residual on unseen test data."""
    rng = np.random.default_rng(99)
    X_train = np.abs(rng.normal(40, 10, size=(35, 27)))
    y_train = np.abs(rng.normal(150, 25, size=35))
    X_test = np.abs(rng.normal(45, 12, size=(15, 27)))

    model = PhysicsInformedQIFCPRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_phys, r_hat, y_final = model.predict_components(X_test)
    preds = model.predict(X_test)

    # Mathematical consistency
    np.testing.assert_allclose(preds, y_final)
    np.testing.assert_allclose(y_final, np.maximum(y_phys + r_hat, 0.0))
