"""Deterministic test suite for CSVDatasetLoader.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1B tests validating CSV file parsing, schema typing, error detection on corrupt files,
and implementation of the DatasetLoader interface contract.
"""

from pathlib import Path
import pytest

from contracts.exceptions import DataValidationError
from contracts.interfaces import DatasetLoader
from contracts.schemas import VoyageRecord
from src.ingestion.dataset_loader import CSVDatasetLoader


def test_loader_implements_interface() -> None:
    """Ensure CSVDatasetLoader conforms to the abstract DatasetLoader contract."""
    loader = CSVDatasetLoader()
    assert isinstance(loader, DatasetLoader)
    assert hasattr(loader, "load_data")
    assert hasattr(loader, "validate_data")


def test_load_sample_csv_dataset() -> None:
    """Verify loading the 10,000-row sample dataset produces typed VoyageRecord instances."""
    sample_csv = Path("data/raw/voyages_sample.csv")
    if not sample_csv.exists():
        pytest.skip("Sample CSV dataset not generated yet.")

    loader = CSVDatasetLoader()
    records = loader.load_data(sample_csv)

    assert len(records) == 10000
    assert all(isinstance(rec, VoyageRecord) for rec in records)

    first = records[0]
    assert first.voyage_id.startswith("VY-")
    assert isinstance(first.vessel_dwt, float)
    assert isinstance(first.sea_state, int)
    assert isinstance(first.is_synthetic, bool)
    assert first.fuel_consumption is not None
    assert first.co2_emissions is not None

    # Verify domain validation succeeds
    assert loader.validate_data(records, strict=False) is True


def test_load_nonexistent_file_raises_error() -> None:
    """Ensure attempting to load a non-existent file raises DataValidationError."""
    loader = CSVDatasetLoader()
    with pytest.raises(DataValidationError) as exc_info:
        loader.load_data("data/raw/does_not_exist.csv")

    assert "does not exist" in str(exc_info.value)


def test_load_directory_raises_error(tmp_path: Path) -> None:
    """Ensure passing a directory path raises DataValidationError."""
    loader = CSVDatasetLoader()
    with pytest.raises(DataValidationError) as exc_info:
        loader.load_data(tmp_path)

    assert "not a file" in str(exc_info.value)


def test_load_empty_file_raises_error(tmp_path: Path) -> None:
    """Ensure loading an empty file raises DataValidationError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    loader = CSVDatasetLoader()
    with pytest.raises(DataValidationError) as exc_info:
        loader.load_data(empty_file)

    assert "empty" in str(exc_info.value)


def test_missing_header_column_raises_error(tmp_path: Path) -> None:
    """Ensure CSV missing mandatory columns raises DataValidationError."""
    corrupt_csv = tmp_path / "missing_cols.csv"
    corrupt_csv.write_text(
        "voyage_id,vessel_id,distance_nm\nVY-001,VSL-001,500.0\n",
        encoding="utf-8",
    )

    loader = CSVDatasetLoader()
    with pytest.raises(DataValidationError) as exc_info:
        loader.load_data(corrupt_csv)

    assert "missing mandatory columns" in str(exc_info.value)


def test_malformed_numeric_field_raises_error(tmp_path: Path) -> None:
    """Ensure malformed numeric value in CSV row raises DataValidationError."""
    malformed_csv = tmp_path / "malformed.csv"
    content = (
        "voyage_id,vessel_id,vessel_type,vessel_dwt,cargo_tons,distance_nm,speed_knots,"
        "hours_at_sea,fuel_type,weather_factor,sea_state,data_source,is_synthetic,"
        "fuel_consumption,co2_emissions\n"
        "VY-001,VSL-001,Bulk Carrier,INVALID_FLOAT,50000.0,2000.0,14.0,142.8,"
        "VLSFO,1.05,2,mock,True,450.0,1400.0\n"
    )
    malformed_csv.write_text(content, encoding="utf-8")

    loader = CSVDatasetLoader()
    with pytest.raises(DataValidationError) as exc_info:
        loader.load_data(malformed_csv)

    assert "malformed" in str(exc_info.value)


def test_load_and_validate_convenience_method(tmp_path: Path) -> None:
    """Ensure load_and_validate returns both records and successful ValidationResult."""
    valid_csv = tmp_path / "valid_test.csv"
    content = (
        "voyage_id,vessel_id,vessel_type,vessel_dwt,cargo_tons,distance_nm,speed_knots,"
        "hours_at_sea,fuel_type,weather_factor,sea_state,data_source,is_synthetic,"
        "fuel_consumption,co2_emissions\n"
        "VY-T1,VSL-001,Bulk Carrier,60000.0,50000.0,2000.0,14.0,142.8,"
        "Diesel,1.05,2,mock,True,450.0,1400.0\n"
        "VY-T2,VSL-002,Container Ship,40000.0,30000.0,1500.0,18.0,83.3,"
        "LNG,1.10,3,mock,True,380.0,1216.0\n"
    )
    valid_csv.write_text(content, encoding="utf-8")

    loader = CSVDatasetLoader()
    records, result = loader.load_and_validate(valid_csv, strict=True)

    assert len(records) == 2
    assert result.is_valid is True
    assert result.total_records == 2
    assert result.valid_records == 2
