"""Contract version compatibility tests.

Validates semantic versioning of API contracts and ensures that all schemas
maintain the required dataclass field structure for backward compatibility.
"""

from dataclasses import FrozenInstanceError
import pytest

from contracts import (
    CONTRACT_VERSION,
    ComplianceResult,
    EmissionResult,
    FleetAssignment,
    FuelType,
    OptimizationResult,
    PredictionResult,
    ScenarioResult,
    VoyageRecord,
)
from contracts.version import CONTRACT_VERSION as DIRECT_VERSION


def test_contract_version_value() -> None:
    """Verify that CONTRACT_VERSION is set to canonical version 1.0.0."""
    assert CONTRACT_VERSION == "1.0.0"
    assert DIRECT_VERSION == "1.0.0"
    parts = CONTRACT_VERSION.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_voyage_record_schema_fields() -> None:
    """Verify VoyageRecord schema contract has all required upgraded fields."""
    assert hasattr(VoyageRecord, "__dataclass_fields__")
    expected_fields = {
        "voyage_id",
        "vessel_id",
        "vessel_type",
        "vessel_dwt",
        "cargo_tons",
        "distance_nm",
        "speed_knots",
        "hours_at_sea",
        "fuel_type",
        "weather_factor",
        "sea_state",
        "data_source",
        "is_synthetic",
        "fuel_consumption",
        "co2_emissions",
    }
    actual_fields = set(VoyageRecord.__dataclass_fields__.keys())
    assert expected_fields == actual_fields


def test_voyage_record_optional_defaults() -> None:
    """Verify that inference-time optional fields default to None."""
    inference_record = VoyageRecord(
        voyage_id="VY-INF-01",
        vessel_id="VSL-Tanker-1",
        vessel_type="Tanker",
        vessel_dwt=115000.0,
        cargo_tons=95000.0,
        distance_nm=3200.0,
        speed_knots=13.2,
        hours_at_sea=242.4,
        fuel_type=FuelType.DIESEL.value,
        weather_factor=1.08,
        sea_state=3,
        data_source="thetis",
        is_synthetic=False,
    )
    assert inference_record.fuel_consumption is None
    assert inference_record.co2_emissions is None
    assert inference_record.vessel_dwt == 115000.0
    assert inference_record.hours_at_sea == 242.4
    assert inference_record.data_source == "thetis"
    assert inference_record.is_synthetic is False


def test_voyage_record_frozen_and_slotted() -> None:
    """Verify that VoyageRecord remains immutable (frozen) and memory-optimized (slotted)."""
    record = VoyageRecord(
        voyage_id="VY-FROZEN",
        vessel_id="VSL-Bulker-1",
        vessel_type="Bulker",
        vessel_dwt=82000.0,
        cargo_tons=75000.0,
        distance_nm=1500.0,
        speed_knots=12.5,
        hours_at_sea=120.0,
        fuel_type=FuelType.LNG.value,
        weather_factor=1.0,
        sea_state=2,
        data_source="synthetic",
        is_synthetic=True,
    )
    # Check slotted
    assert hasattr(VoyageRecord, "__slots__")
    assert "__dict__" not in dir(record)

    # Check frozen
    with pytest.raises(FrozenInstanceError):
        record.speed_knots = 14.0  # type: ignore[misc]


def test_prediction_result_schema_fields() -> None:
    """Verify PredictionResult schema contract has required fields."""
    assert hasattr(PredictionResult, "__dataclass_fields__")
    expected_fields = {
        "model_name",
        "predicted_fuel_consumption",
        "confidence_score",
        "runtime_seconds",
    }
    actual_fields = set(PredictionResult.__dataclass_fields__.keys())
    assert expected_fields.issubset(actual_fields)


def test_emission_result_schema_fields() -> None:
    """Verify EmissionResult schema contract has required fields."""
    assert hasattr(EmissionResult, "__dataclass_fields__")
    expected_fields = {"fuel_type", "co2", "ch4", "n2o", "co2e"}
    actual_fields = set(EmissionResult.__dataclass_fields__.keys())
    assert expected_fields.issubset(actual_fields)


def test_compliance_result_schema_fields() -> None:
    """Verify ComplianceResult schema contract has required fields."""
    assert hasattr(ComplianceResult, "__dataclass_fields__")
    expected_fields = {"cii_rating", "fueleu_pass", "compliance_score"}
    actual_fields = set(ComplianceResult.__dataclass_fields__.keys())
    assert expected_fields.issubset(actual_fields)


def test_fleet_assignment_schema_fields() -> None:
    """Verify FleetAssignment schema contract has required fields."""
    assert hasattr(FleetAssignment, "__dataclass_fields__")
    expected_fields = {
        "vessel_id",
        "cargo_id",
        "assigned",
        "estimated_cost",
        "estimated_fuel",
    }
    actual_fields = set(FleetAssignment.__dataclass_fields__.keys())
    assert expected_fields.issubset(actual_fields)


def test_optimization_result_schema_fields() -> None:
    """Verify OptimizationResult schema contract has required fields."""
    assert hasattr(OptimizationResult, "__dataclass_fields__")
    expected_fields = {
        "optimizer_name",
        "best_score",
        "runtime_seconds",
        "iterations",
        "converged",
    }
    actual_fields = set(OptimizationResult.__dataclass_fields__.keys())
    assert expected_fields.issubset(actual_fields)


def test_scenario_result_schema_fields() -> None:
    """Verify ScenarioResult schema contract has required fields."""
    assert hasattr(ScenarioResult, "__dataclass_fields__")
    expected_fields = {
        "scenario_name",
        "fuel_type",
        "total_cost",
        "total_emissions",
        "fuel_consumption",
    }
    actual_fields = set(ScenarioResult.__dataclass_fields__.keys())
    assert expected_fields.issubset(actual_fields)
