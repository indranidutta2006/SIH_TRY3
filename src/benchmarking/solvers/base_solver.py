"""Base interface and unified candidate evaluation for benchmark optimization solvers.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Ensures fair, apples-to-apples comparison across all optimization solvers
by sharing the exact same objective formulation, physical equations, operational constraints,
and regulatory compliance models.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Final

import numpy as np

from contracts.constants import (
    DEFAULT_EUR_TO_USD_FX_RATE,
    FUEL_PRICES_USD_PER_TON,
    FuelType,
)
from contracts.schemas import (
    BenchmarkResult,
    OptimizationScenario,
)
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.operations.demand_satisfaction_engine import CargoDemandSatisfactionEngine
from src.operations.reliability_engine import ScheduleReliabilityEngine
from src.physics.fuel_physics_engine import MaritimeFuelPhysicsEngine
from src.prediction.emission_engine import MaritimeEmissionEngine

logger = logging.getLogger("maritime_system")

VESSEL_SPECS: Final[dict[str, dict[str, Any]]] = {
    "feeder": {
        "dwt": 12000.0,
        "design_speed_knots": 13.0,
        "admiralty_coeff": 420.0,
        "annual_voyages": 35,
        "daily_charter_usd": 12000.0,
        "capex_usd": 25_000_000.0,
        "max_available": 20,
    },
    "medium": {
        "dwt": 45000.0,
        "design_speed_knots": 14.5,
        "admiralty_coeff": 520.0,
        "annual_voyages": 20,
        "daily_charter_usd": 22000.0,
        "capex_usd": 50_000_000.0,
        "max_available": 15,
    },
    "large": {
        "dwt": 120000.0,
        "design_speed_knots": 15.5,
        "admiralty_coeff": 650.0,
        "annual_voyages": 10,
        "daily_charter_usd": 38000.0,
        "capex_usd": 95_000_000.0,
        "max_available": 10,
    },
}

FUEL_CAPEX_MULTIPLIER: Final[dict[str, float]] = {
    "diesel": 1.00,
    "lng": 1.15,
    "methanol": 1.20,
    "hydrogen": 1.40,
    "ammonia": 1.35,
}


class BaseBenchmarkSolver(ABC):
    """Abstract base class for all benchmark optimization solvers."""

    def __init__(self, solver_name: str) -> None:
        """Initialize benchmark solver with identification name and engines."""
        self.solver_name = solver_name
        self.physics_engine = MaritimeFuelPhysicsEngine()
        self.emission_engine = MaritimeEmissionEngine()
        self.compliance_engine = MaritimeComplianceEngine()
        self.demand_engine = CargoDemandSatisfactionEngine()
        self.reliability_engine = ScheduleReliabilityEngine()
        self.logger = logger

    @abstractmethod
    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Execute benchmark solver against the unified operational scenario.

        Args:
            scenario: Comprehensive OptimizationScenario context.
            max_iterations: Maximum search iterations or generation count.
            seed: Reproducibility random seed.

        Returns:
            BenchmarkResult capturing standardized performance metrics.
        """
        pass

    def evaluate_candidate(
        self,
        x_f: int,
        x_m: int,
        x_l: int,
        fuel_mix: dict[str, int],
        speed_knots: float,
        scenario: OptimizationScenario,
    ) -> dict[str, Any]:
        """Unified candidate evaluation under exact maritime physical and operational constraints.

        Returns a dictionary of raw metrics:
            objective_score, fuel_tons, cost_usd, emissions_tons,
            reliability_score, demand_satisfaction_rate, is_feasible, penalties.
        """
        total_vessels = x_f + x_m + x_l
        fuel_prices = scenario.fuel_prices or FUEL_PRICES_USD_PER_TON
        w_cost, w_fuel, w_emiss = scenario.weights

        # 1. Zero vessels penalty
        if total_vessels == 0:
            return {
                "objective_score": 1e9,
                "fuel_consumption": 0.0,
                "operational_cost": 0.0,
                "emissions": 0.0,
                "reliability_score": 0.0,
                "demand_satisfaction_rate": 0.0,
                "feasible_solution": False,
                "failure_reason": "ZERO_VESSELS",
            }

        # 2. Demand Satisfaction & Fleet Capacity
        ann_capacity = (
            x_f * VESSEL_SPECS["feeder"]["dwt"] * 0.85 * VESSEL_SPECS["feeder"]["annual_voyages"]
            + x_m * VESSEL_SPECS["medium"]["dwt"] * 0.85 * VESSEL_SPECS["medium"]["annual_voyages"]
            + x_l * VESSEL_SPECS["large"]["dwt"] * 0.85 * VESSEL_SPECS["large"]["annual_voyages"]
        )

        effective_demand = scenario.forecasted_demand if scenario.forecasted_demand is not None else scenario.cargo_demand
        dem_metrics = self.demand_engine.evaluate_demand_satisfaction(
            scenario=scenario,
            delivered_cargo=ann_capacity,
        )

        demand_deficit = max(0.0, (effective_demand * scenario.service_level) - ann_capacity)
        demand_penalty = (demand_deficit / max(effective_demand, 1.0)) * 500.0

        # 3. Capital & Charter Costs
        base_capital = (
            x_f * VESSEL_SPECS["feeder"]["capex_usd"] * 0.10
            + x_m * VESSEL_SPECS["medium"]["capex_usd"] * 0.10
            + x_l * VESSEL_SPECS["large"]["capex_usd"] * 0.10
        )

        weighted_capex_mult = sum(
            fuel_mix.get(k, 0) * FUEL_CAPEX_MULTIPLIER.get(k, 1.0)
            for k in ("diesel", "lng", "methanol", "hydrogen", "ammonia")
        ) / total_vessels

        total_capital = base_capital * weighted_capex_mult
        budget_excess = max(0.0, total_capital - scenario.budget)
        budget_penalty = (budget_excess / max(scenario.budget, 1.0)) * 1000.0

        # 4. Green Transition Rate Constraint
        alt_count = sum(fuel_mix.get(k, 0) for k in ("lng", "methanol", "hydrogen", "ammonia"))
        alt_fraction = alt_count / total_vessels
        transition_excess = max(0.0, alt_fraction - scenario.max_transition_rate)
        transition_penalty = transition_excess * 1000.0

        # 5. Speed, Transit Time & Delays
        speed = float(np.clip(speed_knots, 9.0, 18.0))
        transit_hours = (scenario.route_distance / speed) * scenario.weather_factor
        port_delay = max(0.0, (scenario.port_delay_factor - 1.0) * 12.0)
        total_duration = transit_hours + port_delay
        delay_hours = max(0.0, total_duration - scenario.deadline_hours)

        # 6. Fuel & Emissions
        weighted_dwt = (
            x_f * VESSEL_SPECS["feeder"]["dwt"]
            + x_m * VESSEL_SPECS["medium"]["dwt"]
            + x_l * VESSEL_SPECS["large"]["dwt"]
        ) / total_vessels

        weighted_adm = (
            x_f * VESSEL_SPECS["feeder"]["admiralty_coeff"]
            + x_m * VESSEL_SPECS["medium"]["admiralty_coeff"]
            + x_l * VESSEL_SPECS["large"]["admiralty_coeff"]
        ) / total_vessels

        total_fuel = 0.0
        total_emissions = 0.0
        total_bunker_cost = 0.0

        fuel_type_map = {
            "diesel": FuelType.DIESEL.value,
            "lng": FuelType.LNG.value,
            "methanol": FuelType.METHANOL.value,
            "hydrogen": FuelType.HYDROGEN.value,
            "ammonia": FuelType.AMMONIA.value,
        }

        # Representative single-voyage fuel and emissions scaled across active vessels
        for f_key, count in fuel_mix.items():
            if count <= 0:
                continue
            f_type = fuel_type_map.get(f_key, FuelType.DIESEL.value)
            f_price = fuel_prices.get(f_type, 650.0)

            fuel_vsl = self.physics_engine.calculate_fuel_use(
                distance_nm=scenario.route_distance,
                speed_knots=speed,
                cargo_tons=weighted_dwt * 0.85,
                weather_factor=scenario.weather_factor,
                fuel_type=f_type,
                vessel_dwt=weighted_dwt,
                admiralty_coeff=weighted_adm,
            )
            emiss_vsl = float(self.emission_engine.calculate_wtw(fuel_vsl, f_type).co2e)

            total_fuel += fuel_vsl * count
            total_emissions += emiss_vsl * count
            total_bunker_cost += fuel_vsl * count * f_price

        carbon_cost = total_emissions * scenario.carbon_price
        delay_cost = delay_hours * total_vessels * 500.0  # $500/hr delay demurrage
        total_operational_cost = total_capital + total_bunker_cost + carbon_cost + delay_cost

        # 7. Schedule Reliability
        rel_metrics = self.reliability_engine.evaluate_schedule_reliability(
            scenario=scenario,
            simulated_delays=[delay_hours] * total_vessels,
            missed_voyages=0,
            total_voyages=max(1, total_vessels * 4),
        )

        reliability_deficit = max(0.0, scenario.target_reliability - rel_metrics.reliability_score)
        reliability_penalty = reliability_deficit * 20.0

        # 8. Composite Objective Function
        normalized_cost = total_operational_cost / 1_000_000.0
        normalized_fuel = total_fuel / 100.0
        normalized_emiss = total_emissions / 300.0

        composite_score = (
            w_cost * normalized_cost
            + w_fuel * normalized_fuel
            + w_emiss * normalized_emiss
            + demand_penalty
            + budget_penalty
            + transition_penalty
            + reliability_penalty
        )

        is_feasible = (
            demand_deficit <= 1e-4
            and budget_excess <= 1e-4
            and transition_excess <= 1e-4
            and reliability_deficit <= 1e-4
        )

        return {
            "objective_score": round(composite_score, 4),
            "fuel_consumption": round(total_fuel, 2),
            "operational_cost": round(total_operational_cost, 2),
            "emissions": round(total_emissions, 2),
            "reliability_score": round(rel_metrics.reliability_score, 2),
            "demand_satisfaction_rate": round(dem_metrics.demand_satisfaction_rate, 4),
            "feasible_solution": is_feasible,
            "fleet_mix": {**fuel_mix, "feeder": x_f, "medium": x_m, "large": x_l},
            "speed_knots": speed,
            "ann_capacity": round(ann_capacity, 2),
        }
