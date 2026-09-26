"""Fleet Composition Optimizer for maritime green fleet management.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1 Deliverable: Determines the optimal mix of vessel types (feeder, medium, large)
and fuel technologies (diesel, lng, methanol, hydrogen, ammonia) under operational,
budgetary, availability, transition rate, and statutory environmental constraints.
"""

from datetime import UTC, datetime
import logging
import time
from typing import Any, Final

from contracts.constants import (
    DEFAULT_EUR_TO_USD_FX_RATE,
    FUEL_PRICES_USD_PER_TON,
    FuelType,
)
from contracts.schemas import (
    FleetCompositionResult,
    OptimizationScenario,
    OptimizationStatus,
)
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.physics.fuel_physics_engine import MaritimeFuelPhysicsEngine
from src.prediction.emission_engine import MaritimeEmissionEngine

logger = logging.getLogger("maritime_system")

# Standard naval architectural assumptions by vessel size category
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

# Fuel technology capital/charter adjustment multipliers relative to conventional diesel
FUEL_CAPEX_MULTIPLIER: Final[dict[str, float]] = {
    "Diesel": 1.00,
    "LNG": 1.15,
    "Methanol": 1.20,
    "Hydrogen": 1.40,
    "Ammonia": 1.35,
}


class FleetCompositionOptimizer:
    """Optimizes fleet composition: vessel size classes and alternative fuel allocations."""

    def __init__(
        self,
        emission_engine: MaritimeEmissionEngine | None = None,
        physics_engine: MaritimeFuelPhysicsEngine | None = None,
        compliance_engine: MaritimeComplianceEngine | None = None,
    ) -> None:
        """Initialize fleet composition optimizer reusing existing engines."""
        self.emission_engine = emission_engine or MaritimeEmissionEngine()
        self.physics_engine = physics_engine or MaritimeFuelPhysicsEngine()
        self.compliance_engine = compliance_engine or MaritimeComplianceEngine()
        self.logger = logger

    def optimize_composition(
        self,
        scenario: OptimizationScenario,
        max_fleet_size: int = 25,
    ) -> FleetCompositionResult:
        """Determine optimal fleet mix satisfying scenario constraints and minimizing total cost.

        Decision Variables:
            x_f: Feeder vessel count
            x_m: Medium vessel count
            x_l: Large vessel count
            y_D: Diesel vessel count
            y_L: LNG vessel count
            y_M: Methanol vessel count
            y_H: Hydrogen vessel count
            y_A: Ammonia vessel count

        Args:
            scenario: Comprehensive OptimizationScenario context.
            max_fleet_size: Upper limit on total active fleet size.

        Returns:
            FleetCompositionResult with status, fleet mix, costs, emissions, and metadata.
        """
        start_time = time.perf_counter()
        self.logger.info(
            "Starting Fleet Composition Optimization for scenario '%s' (Demand: %.1f tons, Budget: $%.1fM)",
            scenario.scenario_id,
            scenario.cargo_demand,
            scenario.budget / 1e6,
        )

        # 1. Edge Case: Zero or Negative Demand
        if scenario.cargo_demand <= 0.0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return FleetCompositionResult(
                status=OptimizationStatus.SUCCESS,
                fleet_mix={"feeder": 0, "medium": 0, "large": 0, "diesel": 0, "lng": 0, "methanol": 0, "hydrogen": 0, "ammonia": 0},
                total_capacity=0.0,
                fuel_consumption=0.0,
                emissions=0.0,
                operational_cost=0.0,
                carbon_cost=0.0,
                optimization_score=0.0,
                service_level_achieved=1.0,
                metadata={
                    "solver_name": "deterministic_combinatorial_mip",
                    "runtime_ms": round(elapsed_ms, 2),
                    "iterations": 1,
                    "convergence_score": 1.0,
                    "optimization_trace": [{"iteration": 1, "score": 0.0}],
                    "regulatory_breakdown": {"cii_score": 0.0, "fuel_eu_score": 0.0, "compliance_penalty": 0.0},
                },
            )

        fuel_prices = scenario.fuel_prices or FUEL_PRICES_USD_PER_TON
        w_cost, w_fuel, w_emiss = scenario.weights
        effective_demand = scenario.forecasted_demand if scenario.forecasted_demand is not None else scenario.cargo_demand
        target_demand = effective_demand * scenario.service_level

        best_score = float("inf")
        best_candidate: dict[str, Any] | None = None
        infeasible_due_to_budget = False
        infeasible_due_to_demand = True
        optimization_trace: list[dict[str, Any]] = []
        iteration_count = 0

        # Candidate size distributions (feeder, medium, large)
        # Search systematically over realistic fleet integer grids ordered by ascending fleet size
        size_candidates = []
        for x_l in range(0, min(VESSEL_SPECS["large"]["max_available"] + 1, max_fleet_size + 1)):
            for x_m in range(0, min(VESSEL_SPECS["medium"]["max_available"] + 1, max_fleet_size - x_l + 1)):
                for x_f in range(0, min(VESSEL_SPECS["feeder"]["max_available"] + 1, max_fleet_size - x_l - x_m + 1)):
                    tot = x_f + x_m + x_l
                    if tot > 0:
                        size_candidates.append((tot, x_l, x_m, x_f))

        size_candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

        for total_vessels, x_l, x_m, x_f in size_candidates:
            # Total annual throughput capacity
            ann_cap = (
                x_f * VESSEL_SPECS["feeder"]["dwt"] * 0.85 * VESSEL_SPECS["feeder"]["annual_voyages"]
                + x_m * VESSEL_SPECS["medium"]["dwt"] * 0.85 * VESSEL_SPECS["medium"]["annual_voyages"]
                + x_l * VESSEL_SPECS["large"]["dwt"] * 0.85 * VESSEL_SPECS["large"]["annual_voyages"]
            )

            if ann_cap < target_demand:
                continue

            infeasible_due_to_demand = False

            # Base capital / charter costs
            base_fleet_capital = (
                x_f * VESSEL_SPECS["feeder"]["capex_usd"] * 0.10  # 10% annual financing cost
                + x_m * VESSEL_SPECS["medium"]["capex_usd"] * 0.10
                + x_l * VESSEL_SPECS["large"]["capex_usd"] * 0.10
            )

            if base_fleet_capital > scenario.budget:
                infeasible_due_to_budget = True
                continue

            # Branch-and-bound lower bound: if base capital alone exceeds best score, no fuel allocation can beat it
            if w_cost * (base_fleet_capital / 10_000_000.0) >= best_score:
                continue

            # Evaluate fuel technology allocation strategies respecting max_transition_rate
            # Allowed alternative vessels count: floor(total_vessels * max_transition_rate)
            max_alt = int(total_vessels * scenario.max_transition_rate)

            candidate_fuel_mixes = self._generate_fuel_mix_candidates(total_vessels, max_alt)

            # Pre-calculate unit operational metrics across the 5 fuel types for this size configuration
            unit_metrics = {
                f_key: self._evaluate_fleet_operational_costs(
                    x_f, x_m, x_l, {f_key: 1}, scenario, fuel_prices
                )
                for f_key in ("diesel", "lng", "methanol", "hydrogen", "ammonia")
            }

            for fuel_mix in candidate_fuel_mixes:
                iteration_count += 1
                y_D = fuel_mix["diesel"]
                y_L = fuel_mix["lng"]
                y_M = fuel_mix["methanol"]
                y_H = fuel_mix["hydrogen"]
                y_A = fuel_mix["ammonia"]

                # Check transition rate constraint strictly
                alt_count = y_L + y_M + y_H + y_A
                if total_vessels > 0 and (alt_count / total_vessels) > (scenario.max_transition_rate + 1e-6):
                    continue

                # Compute capital outlay adjusted for alternative green powertrains
                weighted_multiplier = (
                    y_D * FUEL_CAPEX_MULTIPLIER["Diesel"]
                    + y_L * FUEL_CAPEX_MULTIPLIER["LNG"]
                    + y_M * FUEL_CAPEX_MULTIPLIER["Methanol"]
                    + y_H * FUEL_CAPEX_MULTIPLIER["Hydrogen"]
                    + y_A * FUEL_CAPEX_MULTIPLIER["Ammonia"]
                ) / total_vessels

                total_capital = base_fleet_capital * weighted_multiplier
                if total_capital > scenario.budget:
                    infeasible_due_to_budget = True
                    continue

                # Linear aggregation of operational metrics across the 5 fuels
                fuel_tons = (
                    y_D * unit_metrics["diesel"][0]
                    + y_L * unit_metrics["lng"][0]
                    + y_M * unit_metrics["methanol"][0]
                    + y_H * unit_metrics["hydrogen"][0]
                    + y_A * unit_metrics["ammonia"][0]
                )
                emissions_tons = (
                    y_D * unit_metrics["diesel"][1]
                    + y_L * unit_metrics["lng"][1]
                    + y_M * unit_metrics["methanol"][1]
                    + y_H * unit_metrics["hydrogen"][1]
                    + y_A * unit_metrics["ammonia"][1]
                )
                fuel_cost = (
                    y_D * unit_metrics["diesel"][2]
                    + y_L * unit_metrics["lng"][2]
                    + y_M * unit_metrics["methanol"][2]
                    + y_H * unit_metrics["hydrogen"][2]
                    + y_A * unit_metrics["ammonia"][2]
                )
                reg_penalty = (
                    y_D * unit_metrics["diesel"][3]
                    + y_L * unit_metrics["lng"][3]
                    + y_M * unit_metrics["methanol"][3]
                    + y_H * unit_metrics["hydrogen"][3]
                    + y_A * unit_metrics["ammonia"][3]
                )

                carbon_cost = emissions_tons * scenario.carbon_price
                total_cost = total_capital + fuel_cost + carbon_cost + reg_penalty

                # Multi-objective scalar
                # Normalize terms for stable comparison: Cost ($10M ref), Fuel (10k t ref), Emissions (30k t ref)
                score = (
                    w_cost * (total_cost / 10_000_000.0)
                    + w_fuel * (fuel_tons / 10_000.0)
                    + w_emiss * (emissions_tons / 30_000.0)
                )

                if iteration_count <= 25 or score < best_score:
                    optimization_trace.append({"iteration": iteration_count, "score": round(score, 4)})

                if score < best_score:
                    best_score = score
                    service_achieved = min(1.0, ann_cap / max(scenario.cargo_demand, 1.0))
                    best_candidate = {
                        "fleet_mix": {
                            "feeder": x_f,
                            "medium": x_m,
                            "large": x_l,
                            "diesel": y_D,
                            "lng": y_L,
                            "methanol": y_M,
                            "hydrogen": y_H,
                            "ammonia": y_A,
                        },
                        "total_capacity": ann_cap,
                        "fuel_consumption": fuel_tons,
                        "emissions": emissions_tons,
                        "operational_cost": total_cost,
                        "carbon_cost": carbon_cost,
                        "optimization_score": score,
                        "service_level_achieved": service_achieved,
                        "reg_breakdown": {
                            "fuel_eu_score": round(reg_penalty, 2),
                            "compliance_penalty": round(reg_penalty, 2),
                            "cii_score": round(min(1.0, (emissions_tons * 1e6) / (ann_cap * scenario.route_distance + 1e-4)), 4),
                        },
                    }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Handle infeasible / constraint violation cases gracefully without raising
        if best_candidate is None:
            if infeasible_due_to_demand:
                status = OptimizationStatus.DEMAND_UNSATISFIABLE
                failure_reason = "DEMAND_NOT_MET"
            elif infeasible_due_to_budget:
                status = OptimizationStatus.BUDGET_EXCEEDED
                failure_reason = "BUDGET_EXCEEDED"
            else:
                status = OptimizationStatus.INFEASIBLE
                failure_reason = "PORT_CONSTRAINT"

            return FleetCompositionResult(
                status=status,
                fleet_mix={"feeder": 0, "medium": 0, "large": 0, "diesel": 0, "lng": 0, "methanol": 0, "hydrogen": 0, "ammonia": 0},
                total_capacity=0.0,
                fuel_consumption=0.0,
                emissions=0.0,
                operational_cost=0.0,
                carbon_cost=0.0,
                optimization_score=float("inf"),
                service_level_achieved=0.0,
                metadata={
                    "solver_name": "deterministic_combinatorial_mip",
                    "runtime_ms": round(elapsed_ms, 2),
                    "iterations": iteration_count,
                    "convergence_score": 0.0,
                    "failure_reason": failure_reason,
                    "optimization_trace": optimization_trace,
                    "regulatory_breakdown": {"cii_score": 0.0, "fuel_eu_score": 0.0, "compliance_penalty": 0.0},
                },
            )

        return FleetCompositionResult(
            status=OptimizationStatus.SUCCESS,
            fleet_mix=best_candidate["fleet_mix"],
            total_capacity=round(best_candidate["total_capacity"], 2),
            fuel_consumption=round(best_candidate["fuel_consumption"], 2),
            emissions=round(best_candidate["emissions"], 2),
            operational_cost=round(best_candidate["operational_cost"], 2),
            carbon_cost=round(best_candidate["carbon_cost"], 2),
            optimization_score=round(best_candidate["optimization_score"], 4),
            service_level_achieved=round(best_candidate["service_level_achieved"], 4),
            metadata={
                "solver_name": "deterministic_combinatorial_mip",
                "runtime_ms": round(elapsed_ms, 2),
                "iterations": iteration_count,
                "convergence_score": 1.0,
                "optimization_trace": optimization_trace[-10:] if len(optimization_trace) > 10 else optimization_trace,
                "regulatory_breakdown": best_candidate["reg_breakdown"],
            },
        )

    def _generate_fuel_mix_candidates(self, total_vessels: int, max_alt: int) -> list[dict[str, int]]:
        """Systematically enumerate all feasible integer fuel allocations across the 5-fuel decision space.

        Governing mathematical formulation:
            y_D + y_L + y_M + y_H + y_A = total_vessels
            y_L + y_M + y_H + y_A <= max_alt
            y_i in Z>=0 for all i in {D, L, M, H, A}
        """
        candidates: list[dict[str, int]] = []
        if total_vessels <= 0:
            return candidates

        bound_alt = min(total_vessels, max(0, max_alt))
        for n_alt in range(bound_alt + 1):
            n_diesel = total_vessels - n_alt
            for y_L in range(n_alt + 1):
                rem_L = n_alt - y_L
                for y_M in range(rem_L + 1):
                    rem_M = rem_L - y_M
                    for y_H in range(rem_M + 1):
                        y_A = rem_M - y_H
                        candidates.append({
                            "diesel": n_diesel,
                            "lng": y_L,
                            "methanol": y_M,
                            "hydrogen": y_H,
                            "ammonia": y_A,
                        })
        return candidates

    def _evaluate_fleet_operational_costs(
        self,
        x_f: int,
        x_m: int,
        x_l: int,
        fuel_mix: dict[str, int],
        scenario: OptimizationScenario,
        fuel_prices: dict[str, float],
    ) -> tuple[float, float, float, float]:
        """Compute cumulative fuel, emissions, bunker cost, and FuelEU compliance penalties."""
        total_vessels = x_f + x_m + x_l
        if total_vessels == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Weighted vessel displacement and speed parameters
        avg_dwt = (
            x_f * VESSEL_SPECS["feeder"]["dwt"]
            + x_m * VESSEL_SPECS["medium"]["dwt"]
            + x_l * VESSEL_SPECS["large"]["dwt"]
        ) / total_vessels

        avg_speed = (
            x_f * VESSEL_SPECS["feeder"]["design_speed_knots"]
            + x_m * VESSEL_SPECS["medium"]["design_speed_knots"]
            + x_l * VESSEL_SPECS["large"]["design_speed_knots"]
        ) / total_vessels

        avg_voyages = (
            x_f * VESSEL_SPECS["feeder"]["annual_voyages"]
            + x_m * VESSEL_SPECS["medium"]["annual_voyages"]
            + x_l * VESSEL_SPECS["large"]["annual_voyages"]
        ) / total_vessels

        cum_fuel = 0.0
        cum_emiss = 0.0
        cum_fuel_cost = 0.0
        cum_reg_penalty = 0.0

        fuel_mapping = {
            "diesel": FuelType.DIESEL.value,
            "lng": FuelType.LNG.value,
            "methanol": FuelType.METHANOL.value,
            "hydrogen": FuelType.HYDROGEN.value,
            "ammonia": FuelType.AMMONIA.value,
        }

        for key, count in fuel_mix.items():
            if count <= 0:
                continue
            f_name = fuel_mapping[key]
            # Hydrodynamic fuel consumption per single voyage
            voyage_fuel = self.physics_engine.calculate_fuel_use(
                distance_nm=scenario.route_distance,
                speed_knots=avg_speed,
                cargo_tons=avg_dwt * 0.8,
                weather_factor=scenario.weather_factor,
                fuel_type=f_name,
                vessel_dwt=avg_dwt,
            )
            annual_vessel_fuel = voyage_fuel * avg_voyages
            fleet_fuel_type = annual_vessel_fuel * count

            # Lifecycle emissions
            emiss_res = self.emission_engine.calculate_wtw(fleet_fuel_type, f_name)
            emiss_val = float(emiss_res.co2e)

            # Bunkering cost
            price_per_ton = fuel_prices.get(f_name, 650.0)
            cost_val = fleet_fuel_type * price_per_ton

            # FuelEU Maritime GHG intensity check
            lcv = 42700.0 if f_name == "Diesel" else (49100.0 if f_name == "LNG" else 19900.0)
            energy_mj = fleet_fuel_type * lcv
            intensity = (emiss_val * 1e6) / energy_mj if energy_mj > 0 else 0.0
            fe_res = self.compliance_engine.evaluate_fueleu(
                ghg_intensity=intensity,
                energy_used_mj=energy_mj,
                year=2025,
            )
            base_penalty_usd = float(fe_res.penalty_eur) * DEFAULT_EUR_TO_USD_FX_RATE
            reg_multiplier = getattr(scenario, "regulation_factor", 1.0)
            penalty_usd = base_penalty_usd * reg_multiplier

            cum_fuel += fleet_fuel_type
            cum_emiss += emiss_val
            cum_fuel_cost += cost_val
            cum_reg_penalty += penalty_usd

        return cum_fuel, cum_emiss, cum_fuel_cost, cum_reg_penalty
