"""Deterministic test suite for FeatureEngineeringPipeline.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1B tests validating maritime domain feature extraction, divide-by-zero safeguards,
strict target leakage prevention, and train/inference matrix consistency.
"""

import numpy as np
import pandas as pd
import pytest

from contracts.exceptions import DataValidationError
from contracts.schemas import VoyageRecord
from src.ingestion.feature_pipeline import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    FeatureEngineeringPipeline,
)


@pytest.fixture
def sample_records() -> list[VoyageRecord]:
    """Fixture providing compliant VoyageRecord instances for testing."""
    return [
        VoyageRecord(
            voyage_id="VY-001",
            vessel_id="VSL-001",
            vessel_type="Bulk Carrier",
            vessel_dwt=100000.0,
            cargo_tons=80000.0,
            distance_nm=2000.0,
            speed_knots=14.0,
            hours_at_sea=142.85,
            fuel_type="Diesel",
            weather_factor=1.1,
            sea_state=3,
            data_source="mock",
            is_synthetic=True,
            fuel_consumption=450.0,
            co2_emissions=1400.0,
        ),
        VoyageRecord(
            voyage_id="VY-002",
            vessel_id="VSL-002",
            vessel_type="Container Ship",
            vessel_dwt=50000.0,
            cargo_tons=40000.0,
            distance_nm=1500.0,
            speed_knots=18.0,
            hours_at_sea=83.33,
            fuel_type="LNG",
            weather_factor=1.2,
            sea_state=4,
            data_source="mock",
            is_synthetic=True,
            fuel_consumption=350.0,
            co2_emissions=962.5,
        ),
    ]


def test_feature_creation_computes_expected_columns(
    sample_records: list[VoyageRecord],
) -> None:
    """Verify engineered physical and kinematic metrics are accurately computed.

    NOTE: implied_speed (D/H) and speed_discrepancy (D/H - V) are intentionally
    absent — both are invertible functions of hours_at_sea and constitute
    duration-proxy leakage.  implied_hours (D/V) is the leakage-safe replacement.
    """
    pipeline = FeatureEngineeringPipeline()
    df = pipeline.create_features(sample_records)

    expected_features = [
        "cargo_ratio",
        "cargo_utilization_pct",
        "transport_work",
        "ton_nautical_miles",
        "power_proxy",
        "implied_hours",          # D/V — operational input, no duration leakage
        "weather_speed_interaction",
        "weather_sea_interaction",
    ]
    for feat in expected_features:
        assert feat in df.columns, f"Missing engineered feature: {feat}"

    # Duration-proxy columns must NOT appear in create_features() output
    assert "implied_speed" not in df.columns, (
        "implied_speed (D/H) must not be in feature matrix — it is a duration proxy."
    )
    assert "speed_discrepancy" not in df.columns, (
        "speed_discrepancy (D/H - V) must not be in feature matrix — it is a duration proxy."
    )

    # Verify physical calculations for record 0
    row0 = df.iloc[0]
    expected_ratio = 80000.0 / 100000.0
    assert pytest.approx(row0["cargo_ratio"], rel=1e-3) == expected_ratio
    assert pytest.approx(row0["cargo_utilization_pct"], rel=1e-3) == expected_ratio * 100.0

    expected_work = 80000.0 * 2000.0
    assert pytest.approx(row0["transport_work"], rel=1e-3) == expected_work

    # Power proxy check: ((0.20 * 100000 + 80000) ** (2/3)) * (14.0 ** 3)
    expected_disp = 0.20 * 100000.0 + 80000.0
    expected_power = (expected_disp ** (2.0 / 3.0)) * (14.0 ** 3.0)
    assert pytest.approx(row0["power_proxy"], rel=1e-3) == expected_power



def test_numerical_stability_zero_division() -> None:
    """Ensure zero values for DWT or speed do not produce NaN or ZeroDivisionError.

    Tests implied_hours (D/V) with zero speed — the 1e-4 clamp must prevent division by zero.
    implied_speed (D/H) and speed_discrepancy are no longer computed, so hours_at_sea=0
    no longer raises a ZeroDivisionError either.
    """
    pipeline = FeatureEngineeringPipeline()
    zero_df = pd.DataFrame(
        [
            {
                "voyage_id": "VY-Z1",
                "vessel_id": "VSL-Z1",
                "vessel_type": "General Cargo",
                "vessel_dwt": 0.0,
                "cargo_tons": 0.0,
                "distance_nm": 100.0,
                "speed_knots": 0.0,   # zero speed — tests 1e-4 clamp in implied_hours
                "hours_at_sea": 0.0,  # zero duration — previously caused D/H division by zero
                "fuel_type": "MGO",
                "weather_factor": 1.0,
                "sea_state": 0,
                "data_source": "mock",
                "is_synthetic": True,
            }
        ]
    )

    featured = pipeline.create_features(zero_df)
    assert not np.isnan(featured["cargo_ratio"].iloc[0])
    assert not np.isinf(featured["cargo_ratio"].iloc[0])
    assert not np.isnan(featured["implied_hours"].iloc[0])
    assert not np.isinf(featured["implied_hours"].iloc[0])
    # Duration-proxy columns must not exist at all
    assert "implied_speed" not in featured.columns
    assert "speed_discrepancy" not in featured.columns



def test_target_leakage_guard_rejects_leakage_columns() -> None:
    """Ensure check_for_target_leakage raises DataValidationError on target leakage."""
    pipeline = FeatureEngineeringPipeline()

    for forbidden_col in FORBIDDEN_LEAKAGE_COLUMNS:
        bad_df = pd.DataFrame({"speed_knots": [14.0], forbidden_col: [100.0]})
        with pytest.raises(DataValidationError) as exc_info:
            pipeline.check_for_target_leakage(bad_df)
        assert "Target leakage violation" in str(exc_info.value)


def test_get_training_features_and_target(sample_records: list[VoyageRecord]) -> None:
    """Verify get_training_features_and_target excludes targets and returns valid X and y."""
    pipeline = FeatureEngineeringPipeline()
    X, y = pipeline.get_training_features_and_target(sample_records, encode_categoricals=True)

    assert len(X) == 2
    assert len(y) == 2
    assert y.name == "fuel_consumption"
    assert pytest.approx(y.iloc[0]) == 450.0

    # Ensure no leakage columns exist in X
    for col in FORBIDDEN_LEAKAGE_COLUMNS:
        assert col not in X.columns, f"Target leakage column {col} leaked into X!"

    # Ensure raw identifiers are dropped
    assert "voyage_id" not in X.columns
    assert "vessel_id" not in X.columns
    assert "data_source" not in X.columns

    # Verify deterministic one-hot columns are present
    assert "vessel_type_bulk_carrier" in X.columns
    assert "fuel_type_diesel" in X.columns


def test_get_inference_features_matches_training_columns(
    sample_records: list[VoyageRecord],
) -> None:
    """Verify inference features match training columns and handle unlabelled telemetry."""
    pipeline = FeatureEngineeringPipeline()
    X_train, _ = pipeline.get_training_features_and_target(
        sample_records, encode_categoricals=True
    )

    # Create unlabelled records (fuel_consumption=None, co2_emissions=None)
    unlabelled_records = [
        VoyageRecord(
            voyage_id="VY-UNL-001",
            vessel_id="VSL-001",
            vessel_type="Bulk Carrier",
            vessel_dwt=100000.0,
            cargo_tons=80000.0,
            distance_nm=2000.0,
            speed_knots=14.0,
            hours_at_sea=142.85,
            fuel_type="Diesel",
            weather_factor=1.1,
            sea_state=3,
            data_source="mock",
            is_synthetic=True,
            fuel_consumption=None,
            co2_emissions=None,
        )
    ]

    X_inf = pipeline.get_inference_features(
        unlabelled_records, encode_categoricals=True
    )

    assert list(X_inf.columns) == list(X_train.columns)
    assert len(X_inf) == 1
    assert "fuel_consumption" not in X_inf.columns
    assert "co2_emissions" not in X_inf.columns
