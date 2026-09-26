"""Data schema definitions for the maritime fuel prediction and fleet optimization system.

Contains strongly-typed dataclasses establishing data contracts between
the data ingestion, prediction, physics, emissions, compliance, scheduling,
and optimization components.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
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
    def rating(self) -> str:
        """Convenience alias for cii_rating."""
        return self.cii_rating

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
    def fueleu_result(self) -> "FuelEUResult | None":
        """Extract dedicated FuelEUResult view, or None if FuelEU was not evaluated.

        Returns None when fueleu_pass is None, indicating this ComplianceResult was
        produced by a standalone CII assessment rather than a FuelEU or combined run.
        Preserves the three-state semantics:

            True   -- FuelEU evaluated, vessel passed
            False  -- FuelEU evaluated, vessel failed
            None   -- FuelEU not evaluated (standalone CII result)

        Previously this property coerced None -> False, converting "not evaluated"
        into "failed" -- a semantic error for any consumer of this compatibility accessor.
        """
        if self.fueleu_pass is None:
            # FuelEU was not part of this assessment -- do not manufacture a failure.
            return None
        return FuelEUResult(
            fueleu_pass=bool(self.fueleu_pass),
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
    energy_consumption_mwh: float = 0.0
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


class OptimizationStatus(StrEnum):
    """Execution outcome status for fleet and routing optimization."""

    SUCCESS = "SUCCESS"
    INFEASIBLE = "INFEASIBLE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    DEMAND_UNSATISFIABLE = "DEMAND_UNSATISFIABLE"
    DEADLINE_VIOLATED = "DEADLINE_VIOLATED"


@dataclass(frozen=True, slots=True)
class OptimizationScenario:
    """Unified operational scenario context passed across all strategy optimizers."""

    cargo_demand: float
    route_distance: float
    deadline_hours: float
    scenario_id: str = "SCENARIO-001"
    created_at: str = ""
    carbon_price: float = 80.0  # USD per metric ton lifecycle CO2e
    budget: float = 100_000_000.0  # Total available fleet capex / charter budget
    weather_factor: float = 1.0  # Weather severity multiplier (>= 1.0)
    vessel_class: str = "PANAMAX"  # FEEDER, PANAMAX, POST_PANAMAX, CAPESIZE
    service_level: float = 0.95  # Target delivery fraction (0.0 to 1.0)
    max_transition_rate: float = 1.0  # Max fraction of fleet allowed to transition fuel technology per cycle
    fuel_prices: dict[str, float] | None = None
    routes: tuple[dict[str, Any], ...] | None = None
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (w_cost, w_fuel, w_emissions)
    target_reliability: float = 90.0  # Target schedule reliability score (0.0 to 100.0)
    port_delay_factor: float = 1.0  # Port congestion / turn-around delay multiplier (>= 1.0)
    forecasted_demand: float | None = None  # Optional forecasted cargo demand
    regulation_factor: float = 1.0  # Regulatory stringency multiplier for compliance penalties (default: 1.0)
    vessel_type: str = "Bulk carrier"  # IMO Resolution MEPC.353(78) statutory vessel category
    capacity_dwt: float | None = None  # Sized vessel deadweight tonnage (DWT)
    annual_voyages: int | None = None  # Annual voyages / round-trips derived from strategy
    annual_distance: float | None = None  # Cumulative annual operating distance (nm)

    @property
    def annual_distance_nm(self) -> float:
        """Cumulative annual distance; falls back to route_distance * annual_voyages."""
        if self.annual_distance is not None and self.annual_distance > 0.0:
            return self.annual_distance
        voyages = self.annual_voyages if (self.annual_voyages is not None and self.annual_voyages > 0) else 20
        return self.route_distance * voyages

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into OptimizationScenario."""
        clean_data = dict(data)
        if "routes" in clean_data and isinstance(clean_data["routes"], list):
            clean_data["routes"] = tuple(clean_data["routes"])
        if "weights" in clean_data and isinstance(clean_data["weights"], list):
            clean_data["weights"] = tuple(clean_data["weights"])
        return cls(**clean_data)


@dataclass(frozen=True, slots=True)
class FleetCompositionResult:
    """Optimal fleet composition decision output under operational constraints."""

    status: OptimizationStatus
    fleet_mix: dict[str, int]
    total_capacity: float
    fuel_consumption: float
    emissions: float
    operational_cost: float
    carbon_cost: float
    optimization_score: float
    service_level_achieved: float
    metadata: dict[str, Any]

    @property
    def optimized_fuel_consumption(self) -> float:
        """Alias for fuel_consumption for backward compatibility."""
        return self.fuel_consumption

    @property
    def optimized_cost(self) -> float:
        """Alias for operational_cost for backward compatibility."""
        return self.operational_cost

    @property
    def optimized_emissions(self) -> float:
        """Alias for emissions for backward compatibility."""
        return self.emissions

    @property
    def num_vessels(self) -> int:
        """Total active vessels in the fleet mix."""
        count = sum(v for k, v in self.fleet_mix.items() if k in {"feeder", "medium", "large"})
        if count <= 0:
            count = max(1, sum(self.fleet_mix.values()))
        return count

    @property
    def fuel_mix(self) -> dict[str, float]:
        """Normalized fuel share proportions."""
        fuel_tokens = ["diesel", "lng", "methanol", "hydrogen", "ammonia"]
        counts = {k: self.fleet_mix.get(k, 0) for k in fuel_tokens if self.fleet_mix.get(k, 0) > 0}
        total = sum(counts.values())
        if total <= 0:
            return {"diesel": 1.0}
        return {k: v / total for k, v in counts.items()}

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class CapacityOptimizationResult:
    """Optimal vessel capacity and sizing recommendation output."""

    status: OptimizationStatus
    vessel_class: str
    recommended_capacity: float
    capacity_teu: float
    utilization_rate: float
    fuel_consumption: float
    cost: float
    emissions: float
    optimal_trips: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class SpeedOptimizationResult:
    """Optimal cruising speed recommendation output."""

    status: OptimizationStatus
    optimal_speed: float
    estimated_eta: float
    fuel_consumption: float
    cost: float
    emissions: float
    delay_hours: float
    metadata: dict[str, Any]

    @property
    def optimal_speed_knots(self) -> float:
        """Alias for optimal_speed for backward compatibility."""
        return self.optimal_speed

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class DemandSatisfactionMetrics:
    """Cargo demand satisfaction evaluation metrics."""

    required_demand: float
    delivered_cargo: float
    demand_satisfaction_rate: float  # 0.0 to 1.0 (capped at 1.0)
    unserved_cargo: float
    service_level_gap: float
    is_satisfied: bool
    status: str = "SATISFIED"

    @property
    def satisfaction_percentage(self) -> float:
        """Demand satisfaction rate represented as percentage (0.0 to 100.0%)."""
        return round(self.demand_satisfaction_rate * 100.0, 2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into DemandSatisfactionMetrics."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ReliabilityMetrics:
    """Schedule reliability and service-level evaluation metrics."""

    reliability_score: float  # 0.0 to 100.0
    on_time_arrival_rate: float  # 0.0 to 1.0
    average_delay_hours: float
    max_delay_hours: float
    missed_voyages: int
    total_voyages: int
    score_breakdown: dict[str, float]  # {"on_time_component": ..., "delay_penalty": ..., "missed_voyage_penalty": ...}
    route_reliability: dict[str, float]  # {route_id: score}
    reliability_trace: tuple[dict[str, Any], ...] = ()

    @property
    def schedule_reliability_score(self) -> float:
        """Alias for compatibility."""
        return self.reliability_score

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["reliability_trace"] = list(self.reliability_trace)
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into ReliabilityMetrics."""
        clean_data = dict(data)
        if "reliability_trace" in clean_data and isinstance(clean_data["reliability_trace"], list):
            clean_data["reliability_trace"] = tuple(clean_data["reliability_trace"])
        return cls(**clean_data)


@dataclass(frozen=True, slots=True)
class FleetStrategyRecommendation:
    """Integrated executive strategy unifying Fleet Mix, Capacity, Speed, and Deployment."""

    status: OptimizationStatus
    scenario: OptimizationScenario
    fleet_mix: FleetCompositionResult
    capacity_recommendation: CapacityOptimizationResult
    speed_recommendation: SpeedOptimizationResult
    deployment_plan: dict[str, list[dict[str, Any]]]
    baseline_comparison: dict[str, dict[str, float]]
    fuel_estimate: float
    cost_estimate: float
    emissions_estimate: float
    service_reliability: float | None = None
    reliability_metrics: ReliabilityMetrics | None = None
    demand_metrics: DemandSatisfactionMetrics | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def composition(self) -> FleetCompositionResult:
        """Alias for fleet_mix for backward compatibility."""
        return self.fleet_mix

    @property
    def speed(self) -> SpeedOptimizationResult:
        """Alias for speed_recommendation for backward compatibility."""
        return self.speed_recommendation

    @property
    def capacity(self) -> CapacityOptimizationResult:
        """Alias for capacity_recommendation for backward compatibility."""
        return self.capacity_recommendation

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return {
            "status": self.status.value,
            "scenario": self.scenario.to_dict(),
            "fleet_mix": self.fleet_mix.to_dict(),
            "capacity_recommendation": self.capacity_recommendation.to_dict(),
            "speed_recommendation": self.speed_recommendation.to_dict(),
            "deployment_plan": self.deployment_plan,
            "baseline_comparison": self.baseline_comparison,
            "fuel_estimate": self.fuel_estimate,
            "cost_estimate": self.cost_estimate,
            "emissions_estimate": self.emissions_estimate,
            "service_reliability": self.service_reliability,
            "reliability_metrics": self.reliability_metrics.to_dict() if self.reliability_metrics else None,
            "demand_metrics": self.demand_metrics.to_dict() if self.demand_metrics else None,
            "summary": self.summary,
        }

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Standardized single-solver benchmark evaluation output."""

    solver_name: str
    runtime_seconds: float
    iterations: int
    objective_score: float
    fuel_consumption: float
    operational_cost: float
    emissions: float
    reliability_score: float
    demand_satisfaction_rate: float
    convergence_score: float
    feasible_solution: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into BenchmarkResult."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """Integrated multi-solver comparative benchmark suite output."""

    scenario: OptimizationScenario
    results: tuple[BenchmarkResult, ...]
    metric_leaders: dict[str, str]  # e.g., {"lowest_fuel": "QPSO", "lowest_cost": "LP", ...}
    comparison_matrix: dict[str, dict[str, float | str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return {
            "scenario": self.scenario.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "metric_leaders": self.metric_leaders,
            "comparison_matrix": self.comparison_matrix,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into BenchmarkSuiteResult."""
        scenario = OptimizationScenario.from_dict(data["scenario"])
        results = tuple(BenchmarkResult.from_dict(r) for r in data["results"])
        return cls(
            scenario=scenario,
            results=results,
            metric_leaders=data.get("metric_leaders", {}),
            comparison_matrix=data.get("comparison_matrix", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class PredictionBenchmarkResult:
    """Predictive model benchmark evaluation metrics."""

    model_name: str
    mae: float
    rmse: float
    r2: float
    inference_time_ms: float
    prediction_bias: float = 0.0
    error_std: float = 0.0
    parameters_count: int | None = None
    is_quantum: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into PredictionBenchmarkResult."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ConvergenceAnalysisResult:
    """Iteration-by-iteration optimization trajectory tracking."""

    solver_name: str
    iterations: tuple[int, ...] = ()
    objective_values: tuple[float, ...] = ()
    best_values: tuple[float, ...] = ()
    improvement_rates: tuple[float, ...] = ()
    total_runtime_seconds: float = 0.0
    final_objective: float = 0.0
    convergence_iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["iterations"] = list(self.iterations)
        d["objective_values"] = list(self.objective_values)
        d["best_values"] = list(self.best_values)
        d["improvement_rates"] = list(self.improvement_rates)
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into ConvergenceAnalysisResult."""
        clean = dict(data)
        for k in ("iterations", "objective_values", "best_values", "improvement_rates"):
            if k in clean and isinstance(clean[k], list):
                clean[k] = tuple(clean[k])
        return cls(**clean)


@dataclass(frozen=True, slots=True)
class ParetoResult:
    """Multi-objective Pareto frontier non-dominated sorting output."""

    scenario_id: str
    solver_name: str
    pareto_points: tuple[dict[str, float], ...] = ()
    dominated_points: tuple[dict[str, float], ...] = ()
    hypervolume: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["pareto_points"] = list(self.pareto_points)
        d["dominated_points"] = list(self.dominated_points)
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into ParetoResult."""
        clean = dict(data)
        if "pareto_points" in clean and isinstance(clean["pareto_points"], list):
            clean["pareto_points"] = tuple(clean["pareto_points"])
        if "dominated_points" in clean and isinstance(clean["dominated_points"], list):
            clean["dominated_points"] = tuple(clean["dominated_points"])
        return cls(**clean)


@dataclass(frozen=True, slots=True)
class StatisticalStabilityResult:
    """Monte Carlo stability statistics across multiple independent random seeds."""

    solver_name: str
    num_seeds: int
    seeds: tuple[int, ...] = ()
    objective_values: tuple[float, ...] = ()
    mean_objective: float = 0.0
    std_objective: float = 0.0
    best_objective: float = 0.0
    worst_objective: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["objective_values"] = list(self.objective_values)
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into StatisticalStabilityResult."""
        clean = dict(data)
        if "seeds" in clean and isinstance(clean["seeds"], list):
            clean["seeds"] = tuple(clean["seeds"])
        if "objective_values" in clean and isinstance(clean["objective_values"], list):
            clean["objective_values"] = tuple(clean["objective_values"])
        return cls(**clean)


@dataclass(frozen=True, slots=True)
class WorkflowBenchmarkResult:
    """End-to-end full maritime system workflow timing and performance."""

    scenario_id: str
    total_workflow_runtime_seconds: float
    step_runtimes_seconds: dict[str, float] = field(default_factory=dict)
    prediction_runtime: float = 0.0
    optimization_runtime: float = 0.0
    reliability_runtime: float = 0.0
    deployment_runtime: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into WorkflowBenchmarkResult."""
        return cls(**data)


# =============================================================================
# PHASE 4: DECISION SUPPORT & STRATEGIC DECARBONIZATION SCHEMAS
# =============================================================================

class EvidenceCategory(str, Enum):
    """Ontological classification of data, parameters, and forecast items."""

    STATUTORY = "STATUTORY"  # Official ratified regulatory requirement (e.g. MEPC.400(83), Reg 2023/1805)
    MODELLED = "MODELLED"    # First-principles physics or established engineering formula
    ASSUMED = "ASSUMED"      # Industry-standard operational assumption (e.g. weather/delay distributions)
    SCENARIO = "SCENARIO"    # Forward-looking exploratory projection (e.g. post-2030 CII extrapolation)


@dataclass(frozen=True, slots=True)
class FuelLifecycleProfile:
    """Detailed Well-to-Wake lifecycle emission and cost profile for specific fuel pathway."""

    fuel_name: str
    production_pathway: str  # e.g., 'grey', 'blue', 'green', 'bio', 'e-fuel', 'fossil'
    production_emission_factor: float  # t CO2e / t fuel
    transport_emission_factor: float   # t CO2e / t fuel
    storage_emission_factor: float     # t CO2e / t fuel (boil-off, slip, fugitives)
    tank_to_wake_factor: float         # t CO2e / t fuel (combustion)
    energy_density_mj_per_ton: float   # LHV in MJ / ton
    renewable_fraction: float = 0.0    # 0.0 to 1.0
    cost_per_ton_usd: float = 650.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into FuelLifecycleProfile."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LifecycleAssessmentResult:
    """Well-to-Wake lifecycle assessment outcome for voyage or fleet profile."""

    fuel_type: str
    pathway: str
    fuel_consumption_tons: float
    tank_to_wake_emissions: float      # Direct combustion emissions (t CO2e)
    well_to_tank_emissions: float      # Total upstream emissions (t CO2e)
    well_to_wake_emissions: float      # Total lifecycle footprint (t CO2e)
    fuel_production_emissions: float   # Upstream feedstock & synthesis emissions (t CO2e)
    fuel_transport_emissions: float    # Upstream transport & bunkering emissions (t CO2e)
    fuel_storage_emissions: float      # Fugitive & boil-off emissions (t CO2e)
    lifecycle_cost: float              # Total fuel + carbon + penalty expenditures (USD)
    energy_content_mj: float           # Total energy delivered (MJ)
    emission_intensity_g_per_mj: float # WTW GHG intensity (g CO2e / MJ)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into LifecycleAssessmentResult."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TransitionMilestone:
    """Single-year fleet composition, fuel mix, and compliance checkpoint in a transition roadmap."""

    year: int
    fleet_mix: dict[str, int]
    fuel_shares: dict[str, float]
    avg_speed_knots: float
    capex_usd: float
    opex_usd: float
    annual_emissions_tons: float
    cii_rating: str
    fueleu_penalty_usd: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransitionRoadmap:
    """Multi-year fleet decarbonization and dual-fuel transition roadmap."""

    roadmap_id: str
    start_year: int
    end_year: int
    milestones: tuple[TransitionMilestone, ...]
    total_transition_capex: float
    total_transition_opex: float
    total_transition_cost: float
    cumulative_emissions_tons: float
    emissions_reduction_pct: float
    regulatory_risk_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["milestones"] = [m.to_dict() if hasattr(m, "to_dict") else m for m in self.milestones]
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into TransitionRoadmap."""
        clean = dict(data)
        if "milestones" in clean and isinstance(clean["milestones"], list):
            clean["milestones"] = tuple(TransitionMilestone(**m) for m in clean["milestones"])
        return cls(**clean)


@dataclass(frozen=True, slots=True)
class RegulatoryForecastResult:
    """Forward compliance trajectory across IMO CII and EU FuelEU Maritime regulations."""

    forecast_id: str
    planning_horizon: tuple[int, ...]
    future_cii_ratings: dict[int, str]        # Year -> Rating ('A'..'E')
    future_fueleu_status: dict[int, str]      # Year -> 'COMPLIANT' / 'DEFICIT'
    projected_penalties_eur: dict[int, float] # Year -> Statutory Penalty EUR
    projected_penalties_usd: dict[int, float] # Year -> Penalty USD
    estimated_compliance_probability: dict[int, float]  # Year -> Estimated prob (0.0 to 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["planning_horizon"] = list(self.planning_horizon)
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into RegulatoryForecastResult."""
        clean = dict(data)
        if "planning_horizon" in clean and isinstance(clean["planning_horizon"], list):
            clean["planning_horizon"] = tuple(clean["planning_horizon"])
        return cls(**clean)


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """Parameterized macro-economic scenario for maritime stress testing."""

    name: str
    description: str
    carbon_price: float
    demand_multiplier: float = 1.0
    fossil_fuel_multiplier: float = 1.0
    alt_fuel_multiplier: float = 1.0
    regulation_factor: float = 1.0
    weather_severity_multiplier: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into ScenarioDefinition."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ScenarioComparisonResult:
    """Multi-scenario sensitivity matrix and ranking outcome."""

    scenario_names: tuple[str, ...]
    metrics_comparison: dict[str, dict[str, float]]
    ranking: tuple[str, ...]
    best_scenario: str
    worst_scenario: str
    sensitivity_analysis: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["scenario_names"] = list(self.scenario_names)
        d["ranking"] = list(self.ranking)
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into ScenarioComparisonResult."""
        clean = dict(data)
        if "scenario_names" in clean and isinstance(clean["scenario_names"], list):
            clean["scenario_names"] = tuple(clean["scenario_names"])
        if "ranking" in clean and isinstance(clean["ranking"], list):
            clean["ranking"] = tuple(clean["ranking"])
        return cls(**clean)


@dataclass(frozen=True, slots=True)
class ExecutiveRecommendation:
    """Executive decision brief synthesizing strategy, investment ROI, and risk management."""

    recommendation_id: str
    recommended_fleet_mix: dict[str, int]
    recommended_fuel_strategy: dict[str, float]
    recommended_cruising_speed: float
    total_investment_capex: float
    annual_net_benefit_usd: float
    roi_percentage: float | None
    payback_years: float | None
    payback_status: str               # e.g., 'OPTIMAL', 'ACCEPTABLE', 'NO_PAYBACK'
    expected_cost_savings_usd: float
    expected_cost_savings_pct: float
    expected_emissions_reduction_tons: float
    expected_emissions_reduction_pct: float
    expected_compliance_benefits: dict[str, Any]
    expected_reliability_impact: dict[str, Any]
    priority_actions: tuple[str, ...]
    risk_and_mitigations: tuple[dict[str, str], ...]
    evidence_items: tuple[dict[str, str], ...]  # [{item, category: STATUTORY/MODELLED/ASSUMED/SCENARIO, notes}]
    executive_summary_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Evidence-tagged itemised economics breakdown (Phase 4 Critical Issue 6).
    # Keys: annual_fuel_savings_usd / _evidence / _note,
    #       annual_carbon_savings_usd / _evidence / _note,
    #       annual_penalty_avoidance_usd / _evidence / _note,
    #       yearly_penalty_avoidance_usd (full annual trajectory),
    #       annual_opex_delta_usd / _evidence / _note,
    #       total_retrofit_capex_usd / _evidence / _source,
    #       annual_net_benefit_usd, investment_horizon_years.
    economics_breakdown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        d["priority_actions"] = list(self.priority_actions)
        d["risk_and_mitigations"] = list(self.risk_and_mitigations)
        d["evidence_items"] = list(self.evidence_items)
        return d

    def to_json(self) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize dictionary into ExecutiveRecommendation."""
        clean = dict(data)
        if "priority_actions" in clean and isinstance(clean["priority_actions"], list):
            clean["priority_actions"] = tuple(clean["priority_actions"])
        if "risk_and_mitigations" in clean and isinstance(clean["risk_and_mitigations"], list):
            clean["risk_and_mitigations"] = tuple(clean["risk_and_mitigations"])
        if "evidence_items" in clean and isinstance(clean["evidence_items"], list):
            clean["evidence_items"] = tuple(clean["evidence_items"])
        return cls(**clean)



