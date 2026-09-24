"""Data schema definitions for the maritime fuel prediction and fleet optimization system.

Contains strongly-typed dataclasses establishing data contracts between
the data ingestion, prediction, physics, emissions, compliance, scheduling,
and optimization components.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Optional, Self


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
class CIIResult:
    """Decoupled statutory IMO Carbon Intensity Indicator (CII) assessment output.

    Attributes:
        cii_rating: Operational letter rating ('A' through 'E').
        attained_cii: Attained operational CII in gCO2 / (Capacity_unit * nm).
        required_cii: Target required CII under IMO MEPC.337(76) & MEPC.400(83).
        cii_ratio: Attained-to-Required CII ratio (< 1.0 indicates outperforming statutory target).
        compliance_status: Standardized compliance status ('COMPLIANT' for A/B/C, 'NON_COMPLIANT' for D/E).
        capacity_metric: Statutory capacity basis ('DWT' or 'GT' under MEPC.353(78) G2).
        reference_line_a: IMO G2 reference line parameter a.
        reference_line_c: IMO G2 reference line parameter c.
        rating_boundaries: IMO MEPC.354(78) G4 boundary vector (exp(d1), exp(d2), exp(d3), exp(d4)).
    """

    cii_rating: str
    attained_cii: float
    required_cii: float
    cii_ratio: float
    compliance_status: str
    capacity_metric: str = "DWT"
    reference_line_a: float = 0.0
    reference_line_c: float = 0.0
    rating_boundaries: tuple[float, float, float, float] = (0.86, 0.94, 1.06, 1.18)

    @property
    def is_compliant(self) -> bool:
        """Return True if rating is Grade A, B, or C."""
        return self.cii_rating in ("A", "B", "C")

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class FuelEUResult:
    """Decoupled EU FuelEU Maritime GHG-intensity compliance and penalty estimation.

    Estimates statutory compliance against Article 4 Well-to-Wake GHG intensity targets
    and Annex IV / Article 23 financial penalties, including the Article 23(2) consecutive-period
    multiplier [1 + (n - 1) / 10].

    Note: This is a GHG-intensity compliance and penalty estimator; it does not evaluate
    Article 5 RFNBO subtargets, Article 6 Onshore Power Supply (OPS) mandates, or pooling/banking.

    Attributes:
        fueleu_pass: Boolean pass/fail against statutory GHG intensity limit.
        fueleu_target: Statutory maximum Well-to-Wake GHG intensity (gCO2eq/MJ) for assessment year.
        ghg_intensity: Attained Well-to-Wake GHG intensity (gCO2eq/MJ).
        penalty_eur: Statutory financial penalty in EUR under EU Regulation (EU) 2023/1805 Article 23.
        compliance_status: Standardized compliance status ('COMPLIANT' or 'NON_COMPLIANT').
        consecutive_deficit_periods: Number of consecutive reporting periods with a compliance deficit.
        penalty_multiplier: Article 23(2) multiplier applied: 1 + (n - 1) / 10.
    """

    fueleu_pass: bool
    fueleu_target: float
    ghg_intensity: float
    penalty_eur: float
    compliance_status: str
    consecutive_deficit_periods: int = 1
    penalty_multiplier: float = 1.0

    @property
    def is_compliant(self) -> bool:
        """Return True if FuelEU intensity is within statutory limits."""
        return self.fueleu_pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class ComplianceAssessment:
    """Unified container for multi-regulatory vessel compliance assessments."""

    cii: Optional[CIIResult] = None
    fueleu: Optional[FuelEUResult] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return {
            "cii": self.cii.to_dict() if self.cii is not None else None,
            "fueleu": self.fueleu.to_dict() if self.fueleu is not None else None,
        }

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class ComplianceResult:
    """Regulatory assessment output (IMO CII letter rating and EU FuelEU status).

    Canonical Schema Attributes:
        cii_rating: Operational letter rating ('A' through 'E' for CII, 'N/A' for FuelEU).
        attained_cii: Attained operational CII in gCO2 / (Capacity_unit * nm).
        required_cii: Target required CII under IMO MEPC.337(76) & MEPC.400(83).
        cii_ratio: Attained-to-Required CII ratio (< 1.0 indicates outperforming statutory target).
        fueleu_pass: Boolean pass/fail against EU FuelEU Maritime statutory GHG intensity limit.
            Note: None when evaluating standalone IMO CII, as FuelEU is not evaluated during CII assessment.
        fueleu_target: Statutory maximum Well-to-Wake GHG intensity (gCO2eq/MJ) for assessment year.
        ghg_intensity: Attained Well-to-Wake GHG intensity (gCO2eq/MJ).
        penalty_eur: Statutory financial penalty in EUR under EU Regulation (EU) 2023/1805 Article 23.
        compliance_status: Standardized compliance status ('COMPLIANT' or 'NON_COMPLIANT').
        compliance_score: DEPRECATED legacy compatibility-only field.
            Warning: Polymorphic across regulations (holds cii_ratio for CII, penalty_eur for FuelEU).
            All internal calculations and downstream optimizers consume `penalty_eur` or `cii_ratio` directly.
        consecutive_deficit_periods: Consecutive deficit periods under FuelEU Article 23(2).
        penalty_multiplier: Article 23(2) deficit multiplier applied to FuelEU penalties.
    """

    cii_rating: str
    attained_cii: float = 0.0
    required_cii: float = 0.0
    cii_ratio: float = 0.0
    fueleu_pass: Optional[bool] = None
    fueleu_target: float = 0.0
    ghg_intensity: float = 0.0
    penalty_eur: float = 0.0
    compliance_status: str = "COMPLIANT"
    compliance_score: float = 0.0  # DEPRECATED: use cii_ratio or penalty_eur
    capacity_metric: str = "DWT"  # Statutory capacity basis ('DWT' or 'GT' under MEPC.353(78) G2)
    reference_line_a: float = 0.0  # IMO G2 curve coefficient a
    reference_line_c: float = 0.0  # IMO G2 curve exponent c
    rating_boundaries: tuple[float, float, float, float] = (0.86, 0.94, 1.06, 1.18)  # IMO MEPC.354(78) G4 boundaries
    consecutive_deficit_periods: int = 1
    penalty_multiplier: float = 1.0

    @property
    def cii_result(self) -> CIIResult:
        """Extract dedicated CIIResult view."""
        return CIIResult(
            cii_rating=self.cii_rating,
            attained_cii=self.attained_cii,
            required_cii=self.required_cii,
            cii_ratio=self.cii_ratio,
            compliance_status=self.compliance_status,
            capacity_metric=self.capacity_metric,
            reference_line_a=self.reference_line_a,
            reference_line_c=self.reference_line_c,
            rating_boundaries=self.rating_boundaries,
        )

    @property
    def fueleu_result(self) -> FuelEUResult:
        """Extract dedicated FuelEUResult view."""
        return FuelEUResult(
            fueleu_pass=bool(self.fueleu_pass) if self.fueleu_pass is not None else False,
            fueleu_target=self.fueleu_target,
            ghg_intensity=self.ghg_intensity,
            penalty_eur=self.penalty_eur,
            compliance_status=self.compliance_status,
            consecutive_deficit_periods=self.consecutive_deficit_periods,
            penalty_multiplier=self.penalty_multiplier,
        )

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
    fuel_cost_usd: float = 0.0
    fueleu_penalty_eur: float = 0.0
    fueleu_penalty_usd: float = 0.0
    exchange_rate_eur_to_usd: float = 1.08

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())
