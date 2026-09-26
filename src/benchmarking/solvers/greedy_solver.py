"""Greedy Fleet Allocation benchmark solver.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Classical heuristic baseline that greedily selects highest-capacity,
lowest-cost vessels to meet cargo demand, with standard eco-speed allocation.
"""

import time
from typing import Any

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver, VESSEL_SPECS


class GreedyFleetSolver(BaseBenchmarkSolver):
    """Deterministic greedy baseline solver for fleet mix and speed allocation."""

    def __init__(self) -> None:
        super().__init__(solver_name="Greedy Allocation")

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Execute greedy fleet sizing and speed assignment."""
        start_time = time.perf_counter()

        target_demand = (
            scenario.forecasted_demand
            if scenario.forecasted_demand is not None
            else scenario.cargo_demand
        ) * scenario.service_level

        # Greedy choice: Prioritize Large vessels (economies of scale), then Medium, then Feeder
        cap_l = VESSEL_SPECS["large"]["dwt"] * 0.85 * VESSEL_SPECS["large"]["annual_voyages"]
        cap_m = VESSEL_SPECS["medium"]["dwt"] * 0.85 * VESSEL_SPECS["medium"]["annual_voyages"]
        cap_f = VESSEL_SPECS["feeder"]["dwt"] * 0.85 * VESSEL_SPECS["feeder"]["annual_voyages"]

        rem = max(0.0, target_demand)
        x_l = min(int(rem // cap_l), VESSEL_SPECS["large"]["max_available"])
        rem -= x_l * cap_l

        x_m = min(int(rem // cap_m) + (1 if rem % cap_m > 0 and rem < cap_m else 0), VESSEL_SPECS["medium"]["max_available"])
        rem -= x_m * cap_m

        x_f = 0
        if rem > 0:
            x_f = min(int(rem // cap_f) + 1, VESSEL_SPECS["feeder"]["max_available"])
            rem -= x_f * cap_f

        # Ensure at least 1 vessel if demand > 0
        total_v = x_l + x_m + x_f
        if total_v == 0 and target_demand > 0:
            x_m = 1
            total_v = 1

        # Greedily allocate alternative fuels up to max_transition_rate
        max_alt = int(total_v * scenario.max_transition_rate)
        n_lng = max_alt // 2
        n_meth = max_alt - n_lng
        n_diesel = total_v - max_alt

        fuel_mix = {
            "diesel": n_diesel,
            "lng": n_lng,
            "methanol": n_meth,
            "hydrogen": 0,
            "ammonia": 0,
        }

        # Greedy speed: Set speed to just meet deadline with 10% safety buffer
        port_delay = max(0.0, (scenario.port_delay_factor - 1.0) * 12.0)
        net_deadline = max(1.0, scenario.deadline_hours - port_delay)
        speed_req = (scenario.route_distance / (net_deadline * 0.90)) * scenario.weather_factor
        speed_knots = float(min(18.0, max(11.0, speed_req)))

        eval_res = self.evaluate_candidate(x_f, x_m, x_l, fuel_mix, speed_knots, scenario)
        elapsed = time.perf_counter() - start_time

        return BenchmarkResult(
            solver_name=self.solver_name,
            runtime_seconds=round(elapsed, 4),
            iterations=1,
            objective_score=eval_res["objective_score"],
            fuel_consumption=eval_res["fuel_consumption"],
            operational_cost=eval_res["operational_cost"],
            emissions=eval_res["emissions"],
            reliability_score=eval_res["reliability_score"],
            demand_satisfaction_rate=eval_res["demand_satisfaction_rate"],
            convergence_score=1.0,
            feasible_solution=eval_res["feasible_solution"],
            metadata={
                "fleet_mix": eval_res["fleet_mix"],
                "speed_knots": eval_res["speed_knots"],
                "ann_capacity": eval_res["ann_capacity"],
            },
        )
