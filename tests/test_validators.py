"""Deterministic test suite for ValidationEngine and ValidationResult.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1B validation tests covering domain bounds, naval architecture constraints,
duplicate voyage detection, and strict validation error raising.
"""

from dataclasses import replace
import pytest

from contracts.exceptions import DataValidationError
from contracts.schemas import VoyageRecord
from src.ingestion.validators import (
    VALID_FUEL_TYPES,
    VALID_VESSEL_TYPES,
    ValidationEngine,
    ValidationResult,
)


@pytest.fixture
def valid_voyage_record() -> VoyageRecord:
    """Fixture providing a standard, physically compliant VoyageRecord."""
    return VoyageRecord(
        voyage_id="VY-TEST-001",
        vessel_id="VSL-001",
        vessel_type="Bulk Carrier",
        vessel_dwt=120000.0,
        cargo_tons=95000.0,
        distance_nm=3500.0,
        speed_knots=13.5,
        hours_at_sea=259.26,
        fuel_type="Diesel",
        weather_factor=1.12,
        sea_state=3,
        data_source="mock",
        is_synthetic=True,
        fuel_consumption=850.5,
        co2_emissions=2648.46,
    )


def test_validation_result_properties() -> None:
    """Validate behavior of ValidationResult dataclass properties."""
    clean_result = ValidationResult(
        total_records=10, valid_records=10, invalid_records=0, validation_errors=[]
    )
    assert clean_result.is_valid is True
    assert clean_result.total_records == 10

    dirty_result = ValidationResult(
        total_records=10,
        valid_records=9,
        invalid_records=1,
        validation_errors=["Error"],
    )
    assert dirty_result.is_valid is False


def test_valid_record_passes(valid_voyage_record: VoyageRecord) -> None:
    """Ensure a compliant VoyageRecord produces zero validation errors."""
    engine = ValidationEngine()
    errors = engine.validate_record(valid_voyage_record)
    assert errors == []


def test_empty_string_fields_flagged(valid_voyage_record: VoyageRecord) -> None:
    """Ensure empty or whitespace mandatory identifiers produce validation errors."""
    engine = ValidationEngine()

    rec_bad_id = replace(valid_voyage_record, voyage_id="   ")
    assert any("voyage_id" in err for err in engine.validate_record(rec_bad_id))

    rec_bad_vessel = replace(valid_voyage_record, vessel_id="")
    assert any("vessel_id" in err for err in engine.validate_record(rec_bad_vessel))

    rec_bad_source = replace(valid_voyage_record, data_source="")
    assert any("data_source" in err for err in engine.validate_record(rec_bad_source))


def test_invalid_categorical_domains(valid_voyage_record: VoyageRecord) -> None:
    """Ensure unapproved vessel types and fuel types produce validation errors."""
    engine = ValidationEngine()

    rec_bad_vessel = replace(valid_voyage_record, vessel_type="Submarine")
    errors_vessel = engine.validate_record(rec_bad_vessel)
    assert any("invalid vessel_type" in err for err in errors_vessel)

    rec_bad_fuel = replace(valid_voyage_record, fuel_type="Kerosene")
    errors_fuel = engine.validate_record(rec_bad_fuel)
    assert any("invalid fuel_type" in err for err in errors_fuel)


def test_dwt_and_cargo_bounds(valid_voyage_record: VoyageRecord) -> None:
    """Ensure payload constraints (DWT > 0, Cargo > 0, Cargo <= DWT) are strictly enforced."""
    engine = ValidationEngine()

    rec_neg_dwt = replace(valid_voyage_record, vessel_dwt=-500.0)
    assert any("vessel_dwt" in err for err in engine.validate_record(rec_neg_dwt))

    rec_zero_cargo = replace(valid_voyage_record, cargo_tons=0.0)
    assert any("cargo_tons" in err for err in engine.validate_record(rec_zero_cargo))

    rec_overload = replace(
        valid_voyage_record, vessel_dwt=50000.0, cargo_tons=70000.0
    )
    assert any("exceeds vessel_dwt" in err for err in engine.validate_record(rec_overload))


def test_kinematic_bounds(valid_voyage_record: VoyageRecord) -> None:
    """Ensure distance, speed, and duration must be strictly positive."""
    engine = ValidationEngine()

    rec_bad_dist = replace(valid_voyage_record, distance_nm=0.0)
    assert any("distance_nm" in err for err in engine.validate_record(rec_bad_dist))

    rec_bad_speed = replace(valid_voyage_record, speed_knots=-2.0)
    assert any("speed_knots" in err for err in engine.validate_record(rec_bad_speed))

    rec_bad_hours = replace(valid_voyage_record, hours_at_sea=-10.0)
    assert any("hours_at_sea" in err for err in engine.validate_record(rec_bad_hours))


def test_environmental_bounds(valid_voyage_record: VoyageRecord) -> None:
    """Ensure weather_factor >= 1.0 and sea_state >= 0."""
    engine = ValidationEngine()

    rec_bad_weather = replace(valid_voyage_record, weather_factor=0.85)
    assert any("weather_factor" in err for err in engine.validate_record(rec_bad_weather))

    rec_bad_sea = replace(valid_voyage_record, sea_state=-1)
    assert any("sea_state" in err for err in engine.validate_record(rec_bad_sea))


def test_target_non_negative(valid_voyage_record: VoyageRecord) -> None:
    """Ensure fuel_consumption and co2_emissions, when present, must be strictly positive."""
    engine = ValidationEngine()

    rec_neg_fuel = replace(valid_voyage_record, fuel_consumption=-10.0)
    assert any("fuel_consumption" in err for err in engine.validate_record(rec_neg_fuel))

    rec_neg_co2 = replace(valid_voyage_record, co2_emissions=0.0)
    assert any("co2_emissions" in err for err in engine.validate_record(rec_neg_co2))


def test_duplicate_voyage_id_detection(valid_voyage_record: VoyageRecord) -> None:
    """Ensure dataset validation identifies duplicate voyage identifiers."""
    engine = ValidationEngine()
    records = [
        valid_voyage_record,
        replace(valid_voyage_record, vessel_id="VSL-002"),  # duplicate voyage_id
    ]

    result = engine.validate_dataset(records, strict=False)
    assert result.is_valid is False
    assert any("Duplicate voyage_id detected" in err for err in result.validation_errors)


def test_strict_mode_raises_data_validation_error(valid_voyage_record: VoyageRecord) -> None:
    """Ensure strict validation raises DataValidationError when any violations are present."""
    engine = ValidationEngine()
    invalid_record = replace(valid_voyage_record, speed_knots=-5.0)

    with pytest.raises(DataValidationError) as exc_info:
        engine.validate_dataset([invalid_record], strict=True)

    assert "Dataset validation failed" in str(exc_info.value)
    assert "speed_knots" in str(exc_info.value)


def test_filter_valid_records(valid_voyage_record: VoyageRecord) -> None:
    """Ensure filter_valid_records isolates compliant records and discards invalid ones."""
    engine = ValidationEngine()
    rec1 = valid_voyage_record
    rec2 = replace(valid_voyage_record, voyage_id="VY-TEST-002", cargo_tons=-100.0)
    rec3 = replace(valid_voyage_record, voyage_id="VY-TEST-003")

    valid_recs, result = engine.filter_valid_records([rec1, rec2, rec3])
    assert len(valid_recs) == 2
    assert result.total_records == 3
    assert result.valid_records == 2
    assert result.invalid_records == 1
    assert valid_recs[0].voyage_id == "VY-TEST-001"
    assert valid_recs[1].voyage_id == "VY-TEST-003"
