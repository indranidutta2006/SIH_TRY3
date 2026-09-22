"""Deterministic unit tests for CarbonEmissionEngine and EmissionResult.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2B tests validating statutory carbon factors (Diesel, LNG, Methanol), zero-carbon fuels,
negative consumption validation, unsupported fuel rejection, and batch operations.
"""

import json
import pytest

from contracts.exceptions import DataValidationError, PredictionError
from contracts.schemas import VoyageRecord
from src.prediction.emission_engine import (
    CarbonEmissionEngine,
    EmissionResult,
    MaritimeEmissionEngine,
)


@pytest.fixture
def sample_voyage() -> VoyageRecord:
    """Fixture providing a standard compliant VoyageRecord."""
    return VoyageRecord(
        voyage_id="VY-EMISS-001",
        vessel_id="VSL-001",
        vessel_type="Bulk Carrier",
        vessel_dwt=120000.0,
        cargo_tons=95000.0,
        distance_nm=3000.0,
        speed_knots=13.0,
        hours_at_sea=230.76,
        fuel_type="Diesel",
        weather_factor=1.1,
        sea_state=3,
        data_source="mock",
        is_synthetic=True,
        fuel_consumption=500.0,
        co2_emissions=1603.0,
    )


def test_diesel_emission_calculation() -> None:
    """Verify statutory Diesel factor of 3.206 is accurately applied."""
    engine = CarbonEmissionEngine()
    result = engine.calculate_emissions(fuel_consumption=100.0, fuel_type="Diesel")

    assert isinstance(result, EmissionResult)
    assert result.fuel_type == "Diesel"
    assert result.emission_factor == 3.206
    assert pytest.approx(result.co2_emissions, rel=1e-4) == 320.6


def test_lng_emission_calculation() -> None:
    """Verify statutory LNG factor of 2.750 is accurately applied."""
    engine = CarbonEmissionEngine()
    result = engine.calculate_emissions(fuel_consumption=200.0, fuel_type="LNG")

    assert result.fuel_type == "LNG"
    assert result.emission_factor == 2.750
    assert pytest.approx(result.co2_emissions, rel=1e-4) == 550.0


def test_methanol_emission_calculation() -> None:
    """Verify statutory Methanol factor of 1.375 is accurately applied."""
    engine = CarbonEmissionEngine()
    result = engine.calculate_emissions(fuel_consumption=100.0, fuel_type="Methanol")

    assert result.fuel_type == "Methanol"
    assert result.emission_factor == 1.375
    assert pytest.approx(result.co2_emissions, rel=1e-4) == 137.5


def test_zero_carbon_fuels() -> None:
    """Verify alternative zero-carbon power sources yield zero operational CO2."""
    engine = CarbonEmissionEngine()
    for fuel in ["Ammonia", "Hydrogen", "ShorePower"]:
        res = engine.calculate_emissions(fuel_consumption=150.0, fuel_type=fuel)
        assert res.emission_factor == 0.0
        assert res.co2_emissions == 0.0


def test_unsupported_fuel_raises_data_validation_error() -> None:
    """Ensure unsupported fuel classifications raise DataValidationError."""
    engine = CarbonEmissionEngine()
    with pytest.raises(DataValidationError) as exc_info:
        engine.calculate_emissions(fuel_consumption=100.0, fuel_type="NuclearFission")

    assert "Unsupported fuel type" in str(exc_info.value)


def test_negative_fuel_raises_data_validation_error() -> None:
    """Ensure negative fuel consumption values are rejected."""
    engine = CarbonEmissionEngine()
    with pytest.raises(DataValidationError) as exc_info:
        engine.calculate_emissions(fuel_consumption=-25.0, fuel_type="Diesel")

    assert "cannot be negative" in str(exc_info.value)


def test_voyage_record_emissions_calculation(sample_voyage: VoyageRecord) -> None:
    """Verify calculating emissions directly from a VoyageRecord instance."""
    engine = CarbonEmissionEngine()
    result = engine.calculate_voyage_emissions(sample_voyage)

    assert result.fuel_consumption == 500.0
    assert pytest.approx(result.co2_emissions, rel=1e-4) == 500.0 * 3.206


def test_voyage_record_null_target_with_override(sample_voyage: VoyageRecord) -> None:
    """Verify calculating emissions for unlabelled voyage using prediction override."""
    unlabelled = VoyageRecord(
        voyage_id=sample_voyage.voyage_id,
        vessel_id=sample_voyage.vessel_id,
        vessel_type=sample_voyage.vessel_type,
        vessel_dwt=sample_voyage.vessel_dwt,
        cargo_tons=sample_voyage.cargo_tons,
        distance_nm=sample_voyage.distance_nm,
        speed_knots=sample_voyage.speed_knots,
        hours_at_sea=sample_voyage.hours_at_sea,
        fuel_type="LNG",
        weather_factor=sample_voyage.weather_factor,
        sea_state=sample_voyage.sea_state,
        data_source=sample_voyage.data_source,
        is_synthetic=sample_voyage.is_synthetic,
        fuel_consumption=None,
        co2_emissions=None,
    )
    engine = CarbonEmissionEngine()

    # Without override, should raise PredictionError
    with pytest.raises(PredictionError):
        engine.calculate_voyage_emissions(unlabelled)

    # With predicted_fuel override, calculates cleanly
    result = engine.calculate_voyage_emissions(unlabelled, predicted_fuel=400.0)
    assert result.fuel_consumption == 400.0
    assert pytest.approx(result.co2_emissions, rel=1e-4) == 400.0 * 2.750


def test_batch_calculate_emissions() -> None:
    """Verify batch emissions computation over multiple voyages."""
    engine = CarbonEmissionEngine()
    fuels = [100.0, 200.0, 300.0]
    types = ["Diesel", "LNG", "Methanol"]

    results = engine.batch_calculate_emissions(fuels, types)
    assert len(results) == 3
    assert pytest.approx(results[0].co2_emissions, rel=1e-4) == 320.6
    assert pytest.approx(results[1].co2_emissions, rel=1e-4) == 550.0
    assert pytest.approx(results[2].co2_emissions, rel=1e-4) == 412.5


def test_batch_length_mismatch_raises_error() -> None:
    """Ensure batch calculation rejects mismatched array lengths."""
    engine = CarbonEmissionEngine()
    with pytest.raises(DataValidationError) as exc_info:
        engine.batch_calculate_emissions([100.0], ["Diesel", "LNG"])
    assert "Mismatched batch inputs" in str(exc_info.value)


def test_emission_result_serialization() -> None:
    """Verify EmissionResult serializes to dict and JSON correctly."""
    engine = MaritimeEmissionEngine()
    res = engine.calculate_emissions(100.0, "Diesel")

    data = res.to_dict()
    assert data["fuel_type"] == "Diesel"
    assert data["co2_emissions"] == 320.6

    json_str = res.to_json()
    parsed = json.loads(json_str)
    assert parsed["fuel_consumption"] == 100.0
