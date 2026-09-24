"""Validation engine enforcing domain, kinematic, and physical bounds on voyage records.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Evaluates dataset integrity, unique identification, naval architectural boundaries,
and maritime categorical domains before ingestion into the feature pipeline.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from typing import Final

from contracts.constants import SUPPORTED_FUELS, FuelType
from contracts.exceptions import DataValidationError
from contracts.schemas import VoyageRecord

# Standard commercial vessel classifications
VALID_VESSEL_TYPES: Final[frozenset[str]] = frozenset(
    {"Bulk Carrier", "Container Ship", "Oil Tanker", "General Cargo"}
)

# Standard marine fuel tokens resolved from contracts
VALID_FUEL_TYPES: Final[frozenset[str]] = frozenset(
    fuel.value for fuel in SUPPORTED_FUELS
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Summary record encapsulating dataset validation metrics and detailed violations."""

    total_records: int
    valid_records: int
    invalid_records: int
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if all evaluated records satisfy validation rules with zero errors."""
        return self.invalid_records == 0 and len(self.validation_errors) == 0


class ValidationEngine:
    """Evaluates individual records and full dataset sequences against maritime domain rules."""

    def __init__(
        self,
        allowed_vessel_types: frozenset[str] = VALID_VESSEL_TYPES,
        allowed_fuel_types: frozenset[str] = VALID_FUEL_TYPES,
    ) -> None:
        """Initialize the validation engine with allowable categories.

        Args:
            allowed_vessel_types: Set of acceptable vessel type strings.
            allowed_fuel_types: Set of acceptable fuel classification strings.
        """
        self.allowed_vessel_types = allowed_vessel_types
        self.allowed_fuel_types = allowed_fuel_types
        self.logger = logging.getLogger("maritime_system")

    def validate_record(self, record: VoyageRecord) -> list[str]:
        """Verify structural, physical, and domain constraints for a single VoyageRecord.

        Args:
            record: VoyageRecord instance to validate.

        Returns:
            List of violation descriptions. Empty list indicates full compliance.
        """
        errors: list[str] = []

        # 1. Identity and Mandatory Strings
        if not record.voyage_id or not record.voyage_id.strip():
            errors.append("Missing mandatory field: voyage_id must be non-empty.")
        if not record.vessel_id or not record.vessel_id.strip():
            errors.append(f"Voyage {record.voyage_id}: vessel_id must be non-empty.")
        if not record.data_source or not record.data_source.strip():
            errors.append(f"Voyage {record.voyage_id}: data_source must be non-empty.")

        # 2. Categorical Domains
        if record.vessel_type not in self.allowed_vessel_types:
            errors.append(
                f"Voyage {record.voyage_id}: invalid vessel_type '{record.vessel_type}'. "
                f"Allowed: {sorted(self.allowed_vessel_types)}."
            )
        if record.fuel_type not in self.allowed_fuel_types:
            errors.append(
                f"Voyage {record.voyage_id}: invalid fuel_type '{record.fuel_type}'. "
                f"Allowed: {sorted(self.allowed_fuel_types)}."
            )

        # 3. Naval Architecture & Payload Bounds
        if record.vessel_dwt <= 0.0:
            errors.append(
                f"Voyage {record.voyage_id}: vessel_dwt ({record.vessel_dwt}) must be > 0."
            )
        if record.cargo_tons <= 0.0:
            errors.append(
                f"Voyage {record.voyage_id}: cargo_tons ({record.cargo_tons}) must be > 0."
            )
        if record.vessel_dwt > 0.0 and record.cargo_tons > record.vessel_dwt:
            errors.append(
                f"Voyage {record.voyage_id}: cargo_tons ({record.cargo_tons}) exceeds "
                f"vessel_dwt ({record.vessel_dwt})."
            )

        # 4. Kinematics and Duration Bounds
        if record.distance_nm <= 0.0:
            errors.append(
                f"Voyage {record.voyage_id}: distance_nm ({record.distance_nm}) must be > 0."
            )
        if record.speed_knots <= 0.0:
            errors.append(
                f"Voyage {record.voyage_id}: speed_knots ({record.speed_knots}) must be > 0."
            )
        if record.hours_at_sea <= 0.0:
            errors.append(
                f"Voyage {record.voyage_id}: hours_at_sea ({record.hours_at_sea}) must be > 0."
            )

        # 5. Environmental Factors (nullable for real telemetry like THETIS-MRV)
        if record.weather_factor is not None and record.weather_factor < 1.0:
            errors.append(
                f"Voyage {record.voyage_id}: weather_factor ({record.weather_factor}) must be >= 1.0."
            )
        if record.sea_state is not None and record.sea_state < 0:
            errors.append(
                f"Voyage {record.voyage_id}: sea_state ({record.sea_state}) must be >= 0."
            )

        # 6. Optional Target Boundaries (when present)
        if record.fuel_consumption is not None and record.fuel_consumption <= 0.0:
            errors.append(
                f"Voyage {record.voyage_id}: fuel_consumption ({record.fuel_consumption}) must be > 0."
            )
        if record.co2_emissions is not None and record.co2_emissions <= 0.0:
            errors.append(
                f"Voyage {record.voyage_id}: co2_emissions ({record.co2_emissions}) must be > 0."
            )

        return errors

    def validate_dataset(
        self,
        records: Sequence[VoyageRecord],
        strict: bool = False,
    ) -> ValidationResult:
        """Audit an entire collection of records for global uniqueness and domain adherence.

        Args:
            records: Sequence of VoyageRecord instances to validate.
            strict: If True, raises DataValidationError when any violations are detected.

        Returns:
            ValidationResult summarizing counts and explicit error messages.

        Raises:
            DataValidationError: If strict is True and validation fails.
        """
        total = len(records)
        all_errors: list[str] = []
        invalid_count = 0

        # 1. Check Global Uniqueness of voyage_id
        voyage_id_counts = Counter(rec.voyage_id for rec in records)
        duplicates = [vid for vid, count in voyage_id_counts.items() if count > 1]
        if duplicates:
            msg = f"Duplicate voyage_id detected for {len(duplicates)} identifier(s): {duplicates[:10]}."
            all_errors.append(msg)
            self.logger.warning(msg)

        # 2. Check Individual Record Bounds
        for rec in records:
            rec_errors = self.validate_record(rec)
            if rec_errors:
                invalid_count += 1
                all_errors.extend(rec_errors)

        valid_count = total - invalid_count

        result = ValidationResult(
            total_records=total,
            valid_records=valid_count,
            invalid_records=invalid_count,
            validation_errors=all_errors,
        )

        self.logger.info(
            "Validation completed: Total=%d | Valid=%d | Invalid=%d | Errors=%d",
            total,
            valid_count,
            invalid_count,
            len(all_errors),
        )

        if strict and not result.is_valid:
            error_preview = "; ".join(all_errors[:5])
            raise DataValidationError(
                f"Dataset validation failed with {len(all_errors)} error(s): {error_preview}",
                details={"errors": all_errors, "total": total, "invalid": invalid_count},
            )

        return result

    def filter_valid_records(
        self,
        records: Sequence[VoyageRecord],
    ) -> tuple[list[VoyageRecord], ValidationResult]:
        """Filter and return only strictly valid records alongside the validation summary.

        Args:
            records: Input sequence of VoyageRecord instances.

        Returns:
            Tuple of (list of valid VoyageRecord instances, ValidationResult).
        """
        # Exclude duplicates first to preserve primary key uniqueness
        voyage_id_counts = Counter(rec.voyage_id for rec in records)
        valid_records: list[VoyageRecord] = []
        all_errors: list[str] = []
        invalid_count = 0

        for rec in records:
            rec_errors = self.validate_record(rec)
            is_duplicate = voyage_id_counts[rec.voyage_id] > 1

            if is_duplicate:
                rec_errors.append(f"Voyage {rec.voyage_id}: duplicate voyage identifier.")

            if rec_errors:
                invalid_count += 1
                all_errors.extend(rec_errors)
            else:
                valid_records.append(rec)

        result = ValidationResult(
            total_records=len(records),
            valid_records=len(valid_records),
            invalid_records=invalid_count,
            validation_errors=all_errors,
        )

        return valid_records, result
