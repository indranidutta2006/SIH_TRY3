"""Voyage optimization recommendation engine for green fleet operations.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2B recommendation system generating deterministic operational interventions
(speed reduction, weather routing, and cargo consolidation) with quantified fuel and CO2 savings.
"""

from dataclasses import asdict, dataclass
import json
import logging
from typing import Any, Final

from contracts.schemas import VoyageRecord
from src.prediction.emission_engine import CarbonEmissionEngine

logger = logging.getLogger("maritime_system")

DEFAULT_FLEET_AVG_SPEED: Final[float] = 14.0  # Commercial merchant fleet benchmark (knots)
WEATHER_MITIGATION_THRESHOLD: Final[float] = 1.15  # Severity factor triggering rerouting
CARGO_EFFICIENCY_THRESHOLD: Final[float] = 0.70  # Capacity utilization threshold (70%)


@dataclass(slots=True, frozen=True)
class OptimizationRecommendation:
    """Structured operational recommendation with quantified fuel and carbon savings."""

    estimated_savings_fuel: float
    estimated_savings_co2: float
    recommended_speed: float
    recommendation_text: str
    opportunities: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=2)


class VoyageOptimizationEngine:
    """Generates rule-based, deterministic decarbonization recommendations for commercial voyages."""

    def __init__(
        self,
        fleet_average_speed: float = DEFAULT_FLEET_AVG_SPEED,
        speed_reduction_pct: float = 0.075,  # 7.5% reduction (within 5-10% range)
        emission_engine: CarbonEmissionEngine | None = None,
    ) -> None:
        """Initialize the optimization engine with operational thresholds.

        Args:
            fleet_average_speed: Speed threshold above which speed reduction is evaluated.
            speed_reduction_pct: Percentage reduction applied to high-speed voyages (default 7.5%).
            emission_engine: CarbonEmissionEngine instance for fuel-to-CO2 translation.
        """
        self.fleet_average_speed = fleet_average_speed
        self.speed_reduction_pct = speed_reduction_pct
        self.emission_engine = emission_engine or CarbonEmissionEngine()
        self.logger = logger

    def evaluate_voyage(
        self,
        voyage: VoyageRecord,
        predicted_fuel_consumption: float,
    ) -> OptimizationRecommendation:
        """Evaluate a voyage and formulate quantitative operational recommendations.

        Rules applied:
        1. Speed Reduction Scenario: If speed > fleet average, recommend a 5-10% reduction.
           Calculates fuel savings using cubic propulsion physics.
        2. Weather Mitigation: If weather_factor > 1.15, advise meteorological rerouting.
        3. Cargo Efficiency: If payload utilization < 70%, suggest consolidation or slow steaming.

        Args:
            voyage: VoyageRecord describing current operational state.
            predicted_fuel_consumption: Predicted base fuel consumption in metric tons.

        Returns:
            OptimizationRecommendation instance.
        """
        if predicted_fuel_consumption <= 0.0:
            return OptimizationRecommendation(
                estimated_savings_fuel=0.0,
                estimated_savings_co2=0.0,
                recommended_speed=voyage.speed_knots,
                recommendation_text="Predicted fuel is zero or invalid; no optimization interventions required.",
                opportunities=[],
            )

        opportunities: list[str] = []
        action_descriptions: list[str] = []

        total_fuel_savings = 0.0
        recommended_speed = voyage.speed_knots

        # 1. Speed Reduction Scenario
        if voyage.speed_knots > self.fleet_average_speed:
            target_speed = round(voyage.speed_knots * (1.0 - self.speed_reduction_pct), 2)
            # Hydrodynamic cubic law: Fuel is proportional to speed^3
            speed_ratio = target_speed / voyage.speed_knots
            fuel_reduction_ratio = 1.0 - (speed_ratio ** 3.0)
            speed_fuel_savings = predicted_fuel_consumption * fuel_reduction_ratio

            total_fuel_savings += speed_fuel_savings
            recommended_speed = target_speed
            opportunities.append("Speed Reduction")
            action_descriptions.append(
                f"Reduce operating speed from {voyage.speed_knots:.1f} to {target_speed:.1f} knots "
                f"({self.speed_reduction_pct * 100:.1f}% reduction). "
                f"Estimated fuel savings: {speed_fuel_savings:.2f} tons."
            )

        # 2. Weather Mitigation Scenario
        if voyage.weather_factor > WEATHER_MITIGATION_THRESHOLD:
            # Weather rerouting can avoid up to 50% of the weather severity penalty
            # Baseline excess resistance = (weather_factor - 1.0)
            weather_penalty = (voyage.weather_factor - 1.0) / voyage.weather_factor
            weather_fuel_savings = predicted_fuel_consumption * (weather_penalty * 0.40)
            total_fuel_savings += weather_fuel_savings
            opportunities.append("Weather Routing")
            action_descriptions.append(
                f"Adverse weather factor ({voyage.weather_factor:.2f} > {WEATHER_MITIGATION_THRESHOLD}) detected. "
                f"Implement optimal weather routing to avoid high sea states. "
                f"Estimated avoidance savings: {weather_fuel_savings:.2f} tons."
            )

        # 3. Cargo Payload Efficiency Scenario
        utilization = (
            (voyage.cargo_tons / voyage.vessel_dwt)
            if voyage.vessel_dwt > 0.0
            else 0.0
        )
        if utilization < CARGO_EFFICIENCY_THRESHOLD:
            opportunities.append("Cargo Consolidation")
            action_descriptions.append(
                f"Low cargo utilization ({utilization * 100:.1f}% < {CARGO_EFFICIENCY_THRESHOLD * 100:.0f}%). "
                "Consider charter consolidation or slow steaming schedule adjustments."
            )

        # Compute corresponding CO2 savings
        if total_fuel_savings > 0.0:
            emission_result = self.emission_engine.calculate_emissions(
                fuel_consumption=total_fuel_savings,
                fuel_type=voyage.fuel_type,
            )
            estimated_co2_savings = emission_result.co2_emissions
        else:
            estimated_co2_savings = 0.0

        if action_descriptions:
            recommendation_text = " | ".join(action_descriptions)
        else:
            recommendation_text = (
                f"Vessel operating within optimal green fleet parameters "
                f"(speed={voyage.speed_knots:.1f} kts, weather_factor={voyage.weather_factor:.2f}, "
                f"cargo_utilization={utilization * 100:.1f}%). No intervention needed."
            )

        return OptimizationRecommendation(
            estimated_savings_fuel=round(total_fuel_savings, 2),
            estimated_savings_co2=round(estimated_co2_savings, 2),
            recommended_speed=round(recommended_speed, 2),
            recommendation_text=recommendation_text,
            opportunities=opportunities,
        )
