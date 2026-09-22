"""Abstract base class contracts defining system integration boundaries.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 0 architectural foundation establishing interfaces for data ingestion, ML prediction,
hydrodynamic fuel physics, greenhouse gas emissions, maritime regulatory compliance,
fleet scheduling, numerical optimization, and scenario simulation.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contracts.schemas import (
    ComplianceResult,
    EmissionResult,
    FleetAssignment,
    OptimizationResult,
    PredictionResult,
    ScenarioResult,
    VoyageRecord,
)


class DatasetLoader(ABC):
    """Abstract contract for data loading, parsing, and data validation components."""

    @abstractmethod
    def load_data(self, source_path: Path | str) -> Sequence[VoyageRecord]:
        """Load maritime voyage records from a persistent storage source.

        Args:
            source_path: Path or URI pointing to raw or processed dataset files.

        Returns:
            Sequence of strongly typed VoyageRecord instances.

        Raises:
            DataValidationError: If dataset format is malformed or inaccessible.
        """
        pass

    @abstractmethod
    def validate_data(self, records: Sequence[VoyageRecord]) -> bool:
        """Validate structural integrity, non-nullability, and domain physical bounds.

        Args:
            records: Sequence of VoyageRecord instances to validate.

        Returns:
            True if all records adhere to validation rules, False otherwise.

        Raises:
            DataValidationError: When boundary or integrity check fails.
        """
        pass


class PredictionEngine(ABC):
    """Abstract contract for machine learning and quantum-inspired predictive engines."""

    @abstractmethod
    def train(
        self,
        training_records: Sequence[VoyageRecord],
        target_field: str = "fuel_consumption",
    ) -> None:
        """Fit predictive model weights using maritime voyage observations.

        Args:
            training_records: Supervised learning records containing features and labels.
            target_field: Field name corresponding to the regression label.

        Raises:
            PredictionError: If training fails or data is insufficient.
        """
        pass

    @abstractmethod
    def predict(self, feature_records: Sequence[VoyageRecord]) -> Sequence[PredictionResult]:
        """Generate fuel consumption estimations for unobserved voyage telemetry.

        Args:
            feature_records: Feature records input for inference.

        Returns:
            Sequence of structured PredictionResult instances.

        Raises:
            PredictionError: If model is uninitialized or inference fails.
        """
        pass

    @abstractmethod
    def evaluate(self, evaluation_records: Sequence[VoyageRecord]) -> dict[str, float]:
        """Calculate statistical regression performance metrics against test records.

        Args:
            evaluation_records: Out-of-sample voyage records with ground truth.

        Returns:
            Dictionary containing error metrics (e.g., RMSE, MAE, R2).

        Raises:
            PredictionError: If ground truth is missing or evaluation fails.
        """
        pass


class FuelPhysicsEngine(ABC):
    """Abstract contract for naval architecture hydrodynamic and energy consumption calculations."""

    @abstractmethod
    def calculate_fuel_use(
        self,
        distance_nm: float,
        speed_knots: float,
        cargo_tons: float,
        weather_factor: float,
        fuel_type: str,
    ) -> float:
        """Calculate total theoretical fuel consumption using hydrodynamic resistance principles.

        Args:
            distance_nm: Voyage distance in nautical miles.
            speed_knots: Vessel operational velocity in knots.
            cargo_tons: Vessel displacement / deadweight payload in metric tons.
            weather_factor: Weather severity multiplier.
            fuel_type: Fuel classification token.

        Returns:
            Computed fuel mass consumption in metric tons.
        """
        pass

    @abstractmethod
    def calculate_energy(
        self,
        fuel_consumption: float,
        fuel_type: str,
    ) -> float:
        """Derive mechanical and thermal propulsion energy in megawatt-hours (MWh).

        Args:
            fuel_consumption: Total fuel consumed in metric tons.
            fuel_type: Fuel classification token denoting Lower Heating Value (LHV).

        Returns:
            Calculated total energy output in megawatt-hours (MWh).
        """
        pass


class EmissionEngine(ABC):
    """Abstract contract for lifecycle greenhouse gas (GHG) accounting."""

    @abstractmethod
    def calculate_ttw(
        self,
        fuel_consumption: float,
        fuel_type: str,
    ) -> EmissionResult:
        """Compute Tank-to-Wake (direct operational combustion) atmospheric emissions.

        Args:
            fuel_consumption: Mass of fuel burned in metric tons.
            fuel_type: Fuel classification token.

        Returns:
            EmissionResult containing CO2, CH4, N2O, and CO2e totals in metric tons.
        """
        pass

    @abstractmethod
    def calculate_wtw(
        self,
        fuel_consumption: float,
        fuel_type: str,
    ) -> EmissionResult:
        """Compute Well-to-Wake (full lifecycle supply chain plus operational) emissions.

        Args:
            fuel_consumption: Mass of fuel consumed in metric tons.
            fuel_type: Fuel classification token.

        Returns:
            EmissionResult quantifying lifecycle emissions across all pollutants.
        """
        pass


class ComplianceEngine(ABC):
    """Abstract contract for maritime decarbonization statutory auditing."""

    @abstractmethod
    def evaluate_cii(
        self,
        co2_emissions: float,
        cargo_tons: float,
        distance_nm: float,
        year: int,
    ) -> ComplianceResult:
        """Calculate IMO Carbon Intensity Indicator (CII) and operational rating (A through E).

        Args:
            co2_emissions: Annual operational CO2 emissions in metric tons.
            cargo_tons: Vessel deadweight tonnage or gross capacity.
            distance_nm: Total distance navigated in nautical miles.
            year: Regulatory compliance reporting calendar year.

        Returns:
            ComplianceResult detailing the categorical letter rating and compliance status.

        Raises:
            ComplianceError: If required vessel rating baselines are undefined.
        """
        pass

    @abstractmethod
    def evaluate_fueleu(
        self,
        ghg_intensity: float,
        energy_used_mj: float,
        year: int,
    ) -> ComplianceResult:
        """Evaluate penalty and compliance status against European Union FuelEU Maritime targets.

        Args:
            ghg_intensity: Attained GHG intensity per energy unit (gCO2eq/MJ).
            energy_used_mj: Total energy consumed on covered voyages in megajoules.
            year: Compliance reporting calendar year.

        Returns:
            ComplianceResult specifying compliance status and regulatory score.

        Raises:
            ComplianceError: If compliance limits for the year cannot be resolved.
        """
        pass


class SchedulerEngine(ABC):
    """Abstract contract for commercial vessel allocation and timetable dispatching."""

    @abstractmethod
    def assign_vessels(
        self,
        vessel_ids: Sequence[str],
        cargo_ids: Sequence[str],
        constraints: dict[str, Any],
    ) -> Sequence[FleetAssignment]:
        """Assign available vessels to pending cargo transport requirements.

        Args:
            vessel_ids: Identifiers of vessels available in fleet inventory.
            cargo_ids: Identifiers of cargo shipments awaiting charter.
            constraints: Operational bounds (berth availability, draft, laycan dates).

        Returns:
            Sequence of FleetAssignment instances representing the optimal allocation.

        Raises:
            SchedulerError: If conflicting constraints render scheduling infeasible.
        """
        pass

    @abstractmethod
    def validate_constraints(
        self,
        assignments: Sequence[FleetAssignment],
        constraints: dict[str, Any],
    ) -> bool:
        """Verify that a candidate fleet assignment schedule respects all physical constraints.

        Args:
            assignments: Candidate list of fleet assignments.
            constraints: Dictionary defining physical, port, and vessel constraints.

        Returns:
            True if all constraints are fully satisfied, False otherwise.
        """
        pass


class OptimizationEngine(ABC):
    """Abstract contract for combinatorial and multi-objective fleet optimization algorithms."""

    @abstractmethod
    def optimize(
        self,
        objective_function: Any,
        parameter_bounds: dict[str, tuple[float, float]],
        hyperparameters: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Execute heuristic or quantum-inspired search to minimize fleet cost/emissions.

        Args:
            objective_function: Callable objective mapping decision vectors to scalar/vector costs.
            parameter_bounds: Dictionary mapping parameter names to (lower_bound, upper_bound).
            hyperparameters: Algorithm-specific configuration settings.

        Returns:
            OptimizationResult containing optimal score, convergence status, and metadata.

        Raises:
            OptimizationError: If optimization diverges or encounters parameter bounds failure.
        """
        pass

    @abstractmethod
    def benchmark(
        self,
        algorithms: Sequence[str],
        test_case: dict[str, Any],
    ) -> dict[str, OptimizationResult]:
        """Execute competitive multi-algorithm benchmark over a standardized problem instance.

        Args:
            algorithms: List of optimizer identifiers to execute (e.g., PSO, QPSO, NSGAII).
            test_case: Standardized test dataset and target formulation.

        Returns:
            Dictionary mapping algorithm identifiers to their respective OptimizationResult.
        """
        pass


class ScenarioEngine(ABC):
    """Abstract contract for macro-level green fleet transition simulation."""

    @abstractmethod
    def run_scenario(
        self,
        scenario_name: str,
        vessel_fleet: Sequence[str],
        operational_parameters: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate fleet operation under an environmental, speed, or alternative fuel regime.

        Args:
            scenario_name: Identifier for the scenario configuration.
            vessel_fleet: List of vessel identifiers evaluated in the run.
            operational_parameters: Environmental, economic, and fuel parameters.

        Returns:
            ScenarioResult summarizing cumulative costs, emissions, and fuel consumption.
        """
        pass

    @abstractmethod
    def compare_scenarios(
        self,
        scenario_results: Sequence[ScenarioResult],
    ) -> list[ScenarioResult]:
        """Rank and analyze tradeoff frontiers across multiple executed scenario runs.

        Args:
            scenario_results: Sequence of executed scenario outcomes.

        Returns:
            Ranked list of ScenarioResult instances sorted by sustainability and cost criteria.
        """
        pass
