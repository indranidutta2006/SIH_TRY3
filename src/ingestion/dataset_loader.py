"""CSV dataset loader implementation adhering to DatasetLoader contract.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1B data ingestion layer providing robust, strongly-typed CSV parsing, type coercion,
and domain validation for maritime voyage records.
"""

from collections.abc import Sequence
import csv
import logging
from pathlib import Path
from typing import Any

from contracts.exceptions import DataValidationError
from contracts.interfaces import DatasetLoader
from contracts.schemas import VoyageRecord
from src.ingestion.validators import ValidationEngine, ValidationResult

logger = logging.getLogger("maritime_system")

REQUIRED_CSV_COLUMNS: frozenset[str] = frozenset(
    {
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
    }
)


def _parse_bool(value: Any) -> bool:
    """Parse string or primitive into boolean value.

    Args:
        value: Input value to parse.

    Returns:
        Boolean representation.

    Raises:
        ValueError: If value cannot be unambiguously coerced to boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "1", "t", "yes", "y"}:
            return True
        if val in {"false", "0", "f", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse '{value}' as boolean.")


def _parse_optional_float(value: Any) -> float | None:
    """Parse optional float field handling empty or NaN representations.

    Args:
        value: Input value to parse.

    Returns:
        Float value or None if field is missing or empty.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"", "none", "nan", "null", "n/a"}:
            return None
        return float(val)
    raise ValueError(f"Cannot parse '{value}' as optional float.")


class CSVDatasetLoader(DatasetLoader):
    """CSV dataset loader implementing the contracts.interfaces.DatasetLoader interface."""

    def __init__(self, validator: ValidationEngine | None = None) -> None:
        """Initialize the CSV dataset loader with an optional validation engine.

        Args:
            validator: ValidationEngine instance. If None, a default instance is created.
        """
        self.validator = validator or ValidationEngine()
        self.logger = logger

    def load_data(self, source_path: Path | str) -> list[VoyageRecord]:
        """Load maritime voyage records from a CSV file into typed VoyageRecord instances.

        Args:
            source_path: Path or string URI pointing to the CSV file.

        Returns:
            List of strongly typed VoyageRecord instances.

        Raises:
            DataValidationError: If file does not exist, is malformed, or missing required columns.
        """
        path = Path(source_path)

        if not path.exists():
            raise DataValidationError(
                f"Dataset source path does not exist: {path.resolve()}",
                details={"source_path": str(path)},
            )
        if not path.is_file():
            raise DataValidationError(
                f"Dataset source path is not a file: {path.resolve()}",
                details={"source_path": str(path)},
            )
        if path.stat().st_size == 0:
            raise DataValidationError(
                f"Dataset source file is empty: {path.resolve()}",
                details={"source_path": str(path)},
            )

        records: list[VoyageRecord] = []
        try:
            with open(path, mode="r", encoding="utf-8-sig", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                if reader.fieldnames is None:
                    raise DataValidationError(
                        f"CSV file has no header columns: {path.resolve()}",
                        details={"source_path": str(path)},
                    )

                header_fields = set(reader.fieldnames)
                missing_columns = REQUIRED_CSV_COLUMNS - header_fields
                if missing_columns:
                    raise DataValidationError(
                        f"CSV missing mandatory columns: {sorted(missing_columns)}",
                        details={"missing_columns": list(missing_columns)},
                    )

                for row_idx, row in enumerate(reader, start=2):
                    try:
                        record = VoyageRecord(
                            voyage_id=str(row["voyage_id"]).strip(),
                            vessel_id=str(row["vessel_id"]).strip(),
                            vessel_type=str(row["vessel_type"]).strip(),
                            vessel_dwt=float(row["vessel_dwt"]),
                            cargo_tons=float(row["cargo_tons"]),
                            distance_nm=float(row["distance_nm"]),
                            speed_knots=float(row["speed_knots"]),
                            hours_at_sea=float(row["hours_at_sea"]),
                            fuel_type=str(row["fuel_type"]).strip(),
                            weather_factor=float(row["weather_factor"]),
                            sea_state=int(row["sea_state"]),
                            data_source=str(row["data_source"]).strip(),
                            is_synthetic=_parse_bool(row["is_synthetic"]),
                            fuel_consumption=_parse_optional_float(
                                row.get("fuel_consumption")
                            ),
                            co2_emissions=_parse_optional_float(
                                row.get("co2_emissions")
                            ),
                        )
                        records.append(record)
                    except (ValueError, KeyError) as parse_err:
                        raise DataValidationError(
                            f"CSV row {row_idx} malformed: {parse_err}",
                            details={"row_index": row_idx, "row_data": row},
                        ) from parse_err

        except (UnicodeDecodeError, csv.Error) as err:
            raise DataValidationError(
                f"Failed to read CSV file {path.resolve()}: {err}",
                details={"source_path": str(path), "error": str(err)},
            ) from err

        self.logger.info(
            "Successfully loaded %d records from %s", len(records), path.name
        )
        return records

    def validate_data(
        self,
        records: Sequence[VoyageRecord],
        strict: bool = False,
    ) -> bool:
        """Validate structural integrity, uniqueness, and maritime domain bounds.

        Args:
            records: Sequence of VoyageRecord instances to validate.
            strict: If True, raises DataValidationError when any errors exist.

        Returns:
            True if all records are valid, False otherwise.

        Raises:
            DataValidationError: If strict is True and validation fails.
        """
        result: ValidationResult = self.validator.validate_dataset(
            records, strict=strict
        )
        return result.is_valid

    def load_and_validate(
        self,
        source_path: Path | str,
        strict: bool = True,
    ) -> tuple[list[VoyageRecord], ValidationResult]:
        """Convenience method to load CSV and validate all records in a single step.

        Args:
            source_path: Path to CSV dataset.
            strict: If True, raises DataValidationError on boundary failures.

        Returns:
            Tuple of (loaded VoyageRecords, ValidationResult).

        Raises:
            DataValidationError: If load fails or strict validation fails.
        """
        records = self.load_data(source_path)
        validation_result = self.validator.validate_dataset(
            records, strict=strict
        )
        return records, validation_result
