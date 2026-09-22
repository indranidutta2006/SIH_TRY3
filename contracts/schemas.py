"""Data schema definitions for the maritime fuel prediction and fleet optimization system.

Contains strongly-typed dataclasses establishing data contracts between
the data ingestion, prediction, physics, emissions, compliance, scheduling,
and optimization components.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class VoyageRecord:
    """Historical or simulated operational voyage telemetry record.

    Represents data ingested from disk, database, or external APIs (e.g. FuelCast, THETIS).
    Provides explicit serialization and deserialization helpers for dataset pipeline I/O.
    """

    voyage_id: str
    vessel_id: str
    vessel_type: str
    vessel_dwt: float
    cargo_tons: float
    distance_nm: float
    speed_knots: float
    hours_at_sea: float
    fuel_type: str
    weather_factor: float
    sea_state: int
    data_source: str
    is_synthetic: bool
    fuel_consumption: float | None = None
    co2_emissions: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into a VoyageRecord instance.

        Args:
            data: Dictionary containing key-value mappings for VoyageRecord fields.

        Returns:
            Instantiated VoyageRecord instance.
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize JSON string into a VoyageRecord instance.

        Args:
            json_str: JSON formatted string representation.

        Returns:
            Instantiated VoyageRecord instance.
        """
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Inference output produced in-memory by a maritime fuel prediction engine."""

    model_name: str
    predicted_fuel_consumption: float
    confidence_score: float
    runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class EmissionResult:
    """Well-to-Wake (WtW) and Tank-to-Wake (TtW) GHG emissions calculation output."""

    fuel_type: str
    co2: float
    ch4: float
    n2o: float
    co2e: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class ComplianceResult:
    """Regulatory assessment output (IMO CII letter rating and EU FuelEU status)."""

    cii_rating: str
    fueleu_pass: bool
    compliance_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class FleetAssignment:
    """Discrete allocation mapping output from the fleet scheduler."""

    vessel_id: str
    cargo_id: str
    assigned: bool
    estimated_cost: float
    estimated_fuel: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Optimization solver output containing convergence telemetry and optimal objective value."""

    optimizer_name: str
    best_score: float
    runtime_seconds: float
    iterations: int
    converged: bool
    history: tuple[float, ...] = ()
    n_evaluations: int = 0
    problem_size: str = ""
    best_vector: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Tradeoff analysis output evaluated across alternative operational scenarios."""

    scenario_name: str
    fuel_type: str
    total_cost: float
    total_emissions: float
    fuel_consumption: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())
