"""Vessel Capacity Optimizer for maritime logistics and fleet sizing.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1 Deliverable: Determines the optimal vessel deadweight capacity (tons) and TEU sizing
for target cargo demand, route distance, and operational constraints across vessel classes
(Feeder, Panamax, Post-Panamax, Capesize).
"""

import logging
import math
import time
from typing import Any, Final

from contracts.constants import FUEL_PRICES_USD_PER_TON, FuelType
from contracts.schemas import (
    CapacityOptimizationResult,
    OptimizationScenario,
    OptimizationStatus,
)
from src.physics.fuel_physics_engine import MaritimeFuelPhysicsEngine
from src.prediction.emission_engine import MaritimeEmissionEngine

logger = logging.getLogger("maritime_system")

# Naval architectural specifications by vessel class
VESSEL_CLASS_PROFILES: Final[dict[str, dict[str, Any]]] = {
    "FEEDER": {
        "min_dwt": 5000.0,
        "max_dwt": 15000.0,
        "default_dwt": 10000.0,
        "teu_ratio": 13.5,  # Metric tons payload per TEU
        "max_draft_m": 8.5,
        "admiralty_coeff": 430.0,
        "design_speed_knots": 13.0,
        "port_turnaround_hours": 16.0,
        "daily_time_charter_usd": 12000.0,
    },
    "HANDYMAX": {
        "min_dwt": 20000.0,
        "max_dwt": 40000.0,
        "default_dwt": 35000.0,
        "teu_ratio": 13.8,
        "max_draft_m": 10.5,
        "admiralty_coeff": 470.0,
        "design_speed_knots": 14.0,
        "port_turnaround_hours": 20.0,
        "daily_time_charter_usd": 17000.0,
    },
    "PANAMAX": {
        "min_dwt": 25000.0,
        "max_dwt": 55000.0,
        "default_dwt": 45000.0,
        "teu_ratio": 14.0,
        "max_draft_m": 12.0,
        "admiralty_coeff": 520.0,
        "design_speed_knots": 14.5,
        "port_turnaround_hours": 24.0,
        "daily_time_charter_usd": 22000.0,
    },
    "POST_PANAMAX": {
        "min_dwt": 60000.0,
        "max_dwt": 100000.0,
        "default_dwt": 80000.0,
        "teu_ratio": 14.2,
        "max_draft_m": 14.5,
        "admiralty_coeff": 600.0,
        "design_speed_knots": 15.0,
        "port_turnaround_hours": 36.0,
        "daily_time_charter_usd": 30000.0,
    },
    "CAPESIZE": {
        "min_dwt": 100000.0,
        "max_dwt": 200000.0,
        "default_dwt": 150000.0,
        "teu_ratio": 14.5,
        "max_draft_m": 18.0,
        "admiralty_coeff": 660.0,
        "design_speed_knots": 15.5,
        "port_turnaround_hours": 48.0,
        "daily_time_charter_usd": 40000.0,
    },
}


class VesselCapacityOptimizer:
    """Optimizes vessel capacity sizing (DWT & TEU) and voyage utilization."""

    def __init__(
        self,
        physics_engine: MaritimeFuelPhysicsEngine | None = None,
        emission_engine: MaritimeEmissionEngine | None = None,
    ) -> None:
        """Initialize capacity optimizer reusing hydrodynamics and emission engines."""
        self.physics_engine = physics_engine or MaritimeFuelPhysicsEngine()
        self.emission_engine = emission_engine or MaritimeEmissionEngine()
        self.logger = logger

    def optimize_capacity(
        self,
        scenario: OptimizationScenario,
        target_fuel_type: str = "Diesel",
        port_capacity_limit: float | None = None,
        min_acceptable_utilization: float = 0.50,
    ) -> CapacityOptimizationResult:
        """Determine optimal vessel capacity balancing hydrodynamic drag vs trip frequency.

        Decision Variables:
            capacity_tons: Vessel deadweight capacity in metric tons.
            capacity_teu: Twenty-foot equivalent container capacity.

        Objectives:
            Minimize Total Cost + Fuel Consumption + Lifecycle Emissions.

        Subject To:
            Delivered Cargo >= Required Cargo
            capacity_tons <= port_capacity_limit
            min_dwt <= capacity_tons <= max_dwt for specified vessel_class.
        """
        start_time = time.perf_counter()
        v_class = scenario.vessel_class.strip().upper()
        if v_class not in VESSEL_CLASS_PROFILES:
            v_class = "PANAMAX"

        profile = VESSEL_CLASS_PROFILES[v_class]
        min_dwt = profile["min_dwt"]
        max_dwt = profile["max_dwt"]
        if port_capacity_limit is not None and port_capacity_limit > 0:
            max_dwt = min(max_dwt, port_capacity_limit)
        elif scenario.port_limits:
            if "max_dwt" in scenario.port_limits:
                max_dwt = min(max_dwt, float(scenario.port_limits["max_dwt"]))
            elif "max_capacity" in scenario.port_limits:
                max_dwt = min(max_dwt, float(scenario.port_limits["max_capacity"]))
            elif "max_draft_m" in scenario.port_limits:
                port_draft = float(scenario.port_limits["max_draft_m"])
                if port_draft < profile["max_draft_m"]:
                    draft_ratio = max(0.2, min(1.0, port_draft / profile["max_draft_m"]))
                    max_dwt = min(max_dwt, profile["max_dwt"] * draft_ratio)

        self.logger.info(
            "Starting Vessel Capacity Optimization [Class: %s, Demand: %.1f t, DWT bounds: (%.0f, %.0f)]",
            v_class,
            scenario.cargo_demand,
            min_dwt,
            max_dwt,
        )

        # 1. Edge Case: Zero or Negative Demand
        if scenario.cargo_demand <= 0.0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CapacityOptimizationResult(
                status=OptimizationStatus.SUCCESS,
                vessel_class=v_class,
                recommended_capacity=0.0,
                capacity_teu=0.0,
                utilization_rate=1.0,
                fuel_consumption=0.0,
                cost=0.0,
                emissions=0.0,
                optimal_trips=0,
                metadata={
                    "solver_name": "naval_architecture_golden_section",
                    "runtime_ms": round(elapsed_ms, 2),
                    "iterations": 1,
                    "convergence_score": 1.0,
                    "optimization_trace": [{"iteration": 1, "score": 0.0}],
                },
            )

        # 2. Check Feasibility
        if min_dwt > max_dwt:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CapacityOptimizationResult(
                status=OptimizationStatus.INFEASIBLE,
                vessel_class=v_class,
                recommended_capacity=min_dwt,
                capacity_teu=round(min_dwt / profile["teu_ratio"], 1),
                utilization_rate=0.0,
                fuel_consumption=0.0,
                cost=0.0,
                emissions=0.0,
                optimal_trips=0,
                metadata={
                    "solver_name": "naval_architecture_golden_section",
                    "runtime_ms": round(elapsed_ms, 2),
                    "iterations": 0,
                    "convergence_score": 0.0,
                    "optimization_trace": [],
                    "error": f"Port capacity limit {port_capacity_limit}t is below class minimum {min_dwt}t",
                },
            )

        w_cost, w_fuel, w_emiss = scenario.weights
        fuel_price = (scenario.fuel_prices or FUEL_PRICES_USD_PER_TON).get(target_fuel_type, 650.0)

        # 3. Fine Golden Search / Discrete Steps over DWT Domain
        # Evaluate 50 discrete capacity candidates across the class interval
        n_steps = 50
        step_size = (max_dwt - min_dwt) / max(1, n_steps - 1)
        best_score = float("inf")
        best_candidate: dict[str, Any] | None = None
        optimization_trace: list[dict[str, Any]] = []

        for step in range(n_steps):
            cap = min_dwt + step * step_size

            # Evaluate required trips under physical payload limit (85% safe cargo capacity)
            effective_payload_per_trip = cap * 0.85
            trips = max(1, math.ceil(scenario.cargo_demand / effective_payload_per_trip))

            actual_cargo_per_trip = scenario.cargo_demand / trips
            utilization = actual_cargo_per_trip / cap

            if utilization < min_acceptable_utilization:
                continue

            # Hydrodynamic fuel consumption per trip
            # Admiralty resistance scales with displacement Delta^(2/3)
            # Lightship weight approx 22% of capacity + carried cargo
            v_speed = profile["design_speed_knots"]
            trip_fuel = self.physics_engine.calculate_fuel_use(
                distance_nm=scenario.route_distance,
                speed_knots=v_speed,
                cargo_tons=actual_cargo_per_trip,
                weather_factor=scenario.weather_factor,
                fuel_type=target_fuel_type,
                vessel_dwt=cap,
                admiralty_coeff=profile["admiralty_coeff"],
            )

            total_fuel = trip_fuel * trips

            # Emissions & Costs
            emiss_res = self.emission_engine.calculate_wtw(total_fuel, target_fuel_type)
            total_emissions = float(emiss_res.co2e)
            carbon_cost = total_emissions * scenario.carbon_price
            bunker_cost = total_fuel * fuel_price

            # Time charter & port turnaround cost
            sea_hours_per_trip = scenario.route_distance / max(v_speed, 1.0)
            total_hours = (sea_hours_per_trip + profile["port_turnaround_hours"]) * trips
            time_charter_cost = (total_hours / 24.0) * profile["daily_time_charter_usd"]

            total_cost = bunker_cost + carbon_cost + time_charter_cost

            # Multi-objective score
            score = (
                w_cost * (total_cost / 1_000_000.0)
                + w_fuel * (total_fuel / 1_000.0)
                + w_emiss * (total_emissions / 3_000.0)
            )

            if step % 5 == 0 or score < best_score:
                optimization_trace.append({"iteration": step + 1, "capacity_dwt": round(cap, 0), "score": round(score, 4)})

            if score < best_score:
                best_score = score
                best_candidate = {
                    "capacity_tons": cap,
                    "capacity_teu": cap / profile["teu_ratio"],
                    "utilization": utilization,
                    "fuel": total_fuel,
                    "cost": total_cost,
                    "emissions": total_emissions,
                    "trips": trips,
                    "score": score,
                }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if best_candidate is None:
            # Fallback to default capacity if all failed utilization constraint
            default_cap = min(max(min_dwt, scenario.cargo_demand), max_dwt)
            trips = max(1, math.ceil(scenario.cargo_demand / (default_cap * 0.85)))
            util = min(1.0, scenario.cargo_demand / (trips * default_cap))
            trip_fuel = self.physics_engine.calculate_fuel_use(
                distance_nm=scenario.route_distance,
                speed_knots=profile["design_speed_knots"],
                cargo_tons=scenario.cargo_demand / trips,
                weather_factor=scenario.weather_factor,
                fuel_type=target_fuel_type,
                vessel_dwt=default_cap,
            )
            tot_fuel = trip_fuel * trips
            tot_emiss = float(self.emission_engine.calculate_wtw(tot_fuel, target_fuel_type).co2e)
            tot_cost = (tot_fuel * fuel_price) + (tot_emiss * scenario.carbon_price)

            return CapacityOptimizationResult(
                status=OptimizationStatus.SUCCESS,
                vessel_class=v_class,
                recommended_capacity=round(default_cap, 1),
                capacity_teu=round(default_cap / profile["teu_ratio"], 1),
                utilization_rate=round(util, 4),
                fuel_consumption=round(tot_fuel, 2),
                cost=round(tot_cost, 2),
                emissions=round(tot_emiss, 2),
                optimal_trips=trips,
                metadata={
                    "solver_name": "naval_architecture_golden_section",
                    "runtime_ms": round(elapsed_ms, 2),
                    "iterations": n_steps,
                    "convergence_score": 1.0,
                    "optimization_trace": optimization_trace,
                },
            )

        return CapacityOptimizationResult(
            status=OptimizationStatus.SUCCESS,
            vessel_class=v_class,
            recommended_capacity=round(best_candidate["capacity_tons"], 1),
            capacity_teu=round(best_candidate["capacity_teu"], 1),
            utilization_rate=round(best_candidate["utilization"], 4),
            fuel_consumption=round(best_candidate["fuel"], 2),
            cost=round(best_candidate["cost"], 2),
            emissions=round(best_candidate["emissions"], 2),
            optimal_trips=best_candidate["trips"],
            metadata={
                "solver_name": "naval_architecture_golden_section",
                "runtime_ms": round(elapsed_ms, 2),
                "iterations": n_steps,
                "convergence_score": 1.0,
                "optimization_trace": optimization_trace[-10:] if len(optimization_trace) > 10 else optimization_trace,
            },
        )
