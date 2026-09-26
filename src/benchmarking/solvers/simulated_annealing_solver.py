"""Simulated Annealing (SA) benchmark solver.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Implements Metropolis-Hastings temperature-annealed neighborhood exploration.
"""

import time
from typing import Any

import numpy as np

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver, VESSEL_SPECS


class SimulatedAnnealingSolver(BaseBenchmarkSolver):
    """Simulated Annealing baseline solver."""

    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.95) -> None:
        super().__init__(solver_name="Simulated Annealing")
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Execute Simulated Annealing search over fleet decisions."""
        start_time = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Dimension 9: [x_f, x_m, x_l, alt_ratio, s_lng, s_methanol, s_hydrogen, s_ammonia, speed]
        lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.5], dtype=float)
        ub = np.array([
            float(VESSEL_SPECS["feeder"]["max_available"]),
            float(VESSEL_SPECS["medium"]["max_available"]),
            float(VESSEL_SPECS["large"]["max_available"]),
            scenario.max_transition_rate,
            1.0,
            1.0,
            1.0,
            1.0,
            17.5,
        ], dtype=float)

        eval_count = 0

        def decode_and_eval(vec: np.ndarray) -> tuple[float, dict[str, Any]]:
            nonlocal eval_count
            eval_count += 1
            xf = int(round(vec[0]))
            xm = int(round(vec[1]))
            xl = int(round(vec[2]))
            tot = xf + xm + xl
            if tot == 0:
                tot = 1
                xm = 1

            alt_ratio = float(np.clip(vec[3], 0.0, scenario.max_transition_rate))
            alt_count = int(round(tot * alt_ratio))
            shares = [float(vec[4]), float(vec[5]), float(vec[6]), float(vec[7])]
            speed = float(np.clip(vec[8], 9.5, 17.5))
            fuel_mix = self.allocate_fuel_mix(tot, alt_count, scenario, shares=shares)
            res = self.evaluate_candidate(xf, xm, xl, fuel_mix, speed, scenario)
            return res["objective_score"], res

        # Initial state
        current_state = rng.uniform(lb, ub)
        # Ensure initial speed is viable for deadline
        viable_speed = scenario.route_distance / max(1.0, scenario.deadline_hours - 12.0) * scenario.weather_factor * 1.05
        current_state[8] = float(np.clip(viable_speed, 11.0, 17.0))

        current_score, current_dict = decode_and_eval(current_state)

        best_state = current_state.copy()
        best_score = current_score
        best_eval_dict = current_dict

        history: list[float] = []
        temp = self.initial_temp

        for _ in range(max_iterations):
            # Generate candidate neighbor
            neighbor = current_state.copy()
            dim_to_perturb = rng.choice(len(lb))

            if dim_to_perturb in (0, 1, 2):
                step = rng.choice([-1.0, 1.0])
                neighbor[dim_to_perturb] = np.clip(neighbor[dim_to_perturb] + step, lb[dim_to_perturb], ub[dim_to_perturb])
            elif dim_to_perturb == 3:
                neighbor[3] = np.clip(neighbor[3] + rng.normal(0.0, 0.05), lb[3], ub[3])
            elif 4 <= dim_to_perturb <= 7:
                neighbor[dim_to_perturb] = np.clip(neighbor[dim_to_perturb] + rng.normal(0.0, 0.1), lb[dim_to_perturb], ub[dim_to_perturb])
            else:
                neighbor[8] = np.clip(neighbor[8] + rng.normal(0.0, 0.5), lb[8], ub[8])

            neighbor_score, neighbor_dict = decode_and_eval(neighbor)
            delta_e = neighbor_score - current_score

            # Metropolis acceptance criterion
            if delta_e <= 0.0 or rng.uniform() < np.exp(-delta_e / max(temp, 1e-6)):
                current_state = neighbor
                current_score = neighbor_score
                current_dict = neighbor_dict

                if current_score < best_score:
                    best_score = current_score
                    best_state = current_state.copy()
                    best_eval_dict = current_dict

            history.append(best_score)
            temp *= self.cooling_rate

        elapsed = time.perf_counter() - start_time
        assert best_eval_dict is not None

        initial_obj = history[0] if history else best_score
        final_obj = history[-1] if history else best_score
        conv_info = {
            "initial_objective": round(float(initial_obj), 4),
            "final_objective": round(float(final_obj), 4),
            "improvement_pct": round(float(initial_obj - final_obj) / max(abs(initial_obj), 1e-4) * 100.0, 2),
            "history": history,
            "n_evaluations": eval_count,
        }

        return BenchmarkResult(
            solver_name=self.solver_name,
            runtime_seconds=round(elapsed, 4),
            iterations=len(history),
            objective_score=best_eval_dict["objective_score"],
            fuel_consumption=best_eval_dict["fuel_consumption"],
            operational_cost=best_eval_dict["operational_cost"],
            emissions=best_eval_dict["emissions"],
            reliability_score=best_eval_dict["reliability_score"],
            demand_satisfaction_rate=best_eval_dict["demand_satisfaction_rate"],
            convergence_score=round(float(initial_obj - final_obj) / max(abs(initial_obj), 1e-4), 4),
            feasible_solution=best_eval_dict["feasible_solution"],
            n_evaluations=eval_count,
            convergence_information=conv_info,
            metadata={
                "fleet_mix": best_eval_dict["fleet_mix"],
                "speed_knots": best_eval_dict["speed_knots"],
                "history": history,
            },
        )
