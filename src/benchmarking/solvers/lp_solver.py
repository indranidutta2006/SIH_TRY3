"""Linear Programming (LP) Relaxation benchmark solver.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Solves the continuous relaxation of the fleet capacity and budget constraints,
projecting onto feasible integer bounds and solving eco-speed.
"""

import time
from typing import Any

import numpy as np
from scipy.optimize import linprog

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver, VESSEL_SPECS


class LinearProgrammingSolver(BaseBenchmarkSolver):
    """Linear Programming continuous relaxation with integer projection solver."""

    def __init__(self) -> None:
        super().__init__(solver_name="Linear Programming (Relaxed)")

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Solve continuous LP relaxation and project onto discrete feasible fleet mix."""
        start_time = time.perf_counter()

        target_demand = (
            scenario.forecasted_demand
            if scenario.forecasted_demand is not None
            else scenario.cargo_demand
        ) * scenario.service_level

        # Objective: minimize annual fleet capital cost
        # c^T x = [c_f, c_m, c_l]
        c = np.array([
            VESSEL_SPECS["feeder"]["capex_usd"] * 0.10,
            VESSEL_SPECS["medium"]["capex_usd"] * 0.10,
            VESSEL_SPECS["large"]["capex_usd"] * 0.10,
        ], dtype=float)

        # Constraint: -Capacity * x <= -Target_Demand
        cap_f = VESSEL_SPECS["feeder"]["dwt"] * 0.85 * VESSEL_SPECS["feeder"]["annual_voyages"]
        cap_m = VESSEL_SPECS["medium"]["dwt"] * 0.85 * VESSEL_SPECS["medium"]["annual_voyages"]
        cap_l = VESSEL_SPECS["large"]["dwt"] * 0.85 * VESSEL_SPECS["large"]["annual_voyages"]

        A_ub = np.array([
            [-cap_f, -cap_m, -cap_l],  # Demand fulfillment
            [c[0], c[1], c[2]],        # Budget ceiling
        ], dtype=float)

        b_ub = np.array([
            -max(0.0, target_demand),
            scenario.budget,
        ], dtype=float)

        bounds = [
            (0.0, float(VESSEL_SPECS["feeder"]["max_available"])),
            (0.0, float(VESSEL_SPECS["medium"]["max_available"])),
            (0.0, float(VESSEL_SPECS["large"]["max_available"])),
        ]

        res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res_lp.success:
            cont_x = res_lp.x
        else:
            # Fallback projection if unbounded/infeasible
            cont_x = np.array([1.0, 1.0, 1.0])

        # Integer rounding projection
        xf = int(np.ceil(cont_x[0]))
        xm = int(np.ceil(cont_x[1]))
        xl = int(np.ceil(cont_x[2]))

        tot = xf + xm + xl
        if tot == 0 and target_demand > 0:
            xm = 1
            tot = 1

        # Allocate allowable alternative fuel up to max_transition_rate
        max_alt = int(tot * scenario.max_transition_rate)
        fuel_mix = self.allocate_fuel_mix(tot, max_alt, scenario)

        # Hydrodynamic eco-speed
        port_delay = max(0.0, (scenario.port_delay_factor - 1.0) * 12.0)
        net_deadline = max(1.0, scenario.deadline_hours - port_delay)
        speed = float(np.clip(scenario.route_distance / (net_deadline * 0.95), 10.0, 16.5))

        eval_res = self.evaluate_candidate(xf, xm, xl, fuel_mix, speed, scenario)
        elapsed = time.perf_counter() - start_time

        nit = int(res_lp.nit) if hasattr(res_lp, "nit") and res_lp.nit else 1
        conv_info = {
            "initial_objective": eval_res["objective_score"],
            "final_objective": eval_res["objective_score"],
            "improvement_pct": 0.0,
            "history": [eval_res["objective_score"]],
            "n_evaluations": 1,
            "lp_iterations": nit,
        }

        return BenchmarkResult(
            solver_name=self.solver_name,
            runtime_seconds=round(elapsed, 4),
            iterations=nit,
            objective_score=eval_res["objective_score"],
            fuel_consumption=eval_res["fuel_consumption"],
            operational_cost=eval_res["operational_cost"],
            emissions=eval_res["emissions"],
            reliability_score=eval_res["reliability_score"],
            demand_satisfaction_rate=eval_res["demand_satisfaction_rate"],
            convergence_score=1.0,
            feasible_solution=eval_res["feasible_solution"],
            n_evaluations=1,
            convergence_information=conv_info,
            metadata={
                "fleet_mix": eval_res["fleet_mix"],
                "speed_knots": eval_res["speed_knots"],
                "lp_status": res_lp.message if hasattr(res_lp, "message") else "solved",
            },
        )
