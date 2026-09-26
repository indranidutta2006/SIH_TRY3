"""Classical Particle Swarm Optimization (PSO) benchmark solver.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Implements velocity-clamped inertia PSO baseline with iteration tracking.
"""

import time
from typing import Any

import numpy as np

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver, VESSEL_SPECS


class ClassicalPSOSolver(BaseBenchmarkSolver):
    """Classical Particle Swarm Optimization baseline solver."""

    def __init__(self, population_size: int = 20) -> None:
        super().__init__(solver_name="Classical PSO")
        self.population_size = population_size

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Execute classical PSO search over fleet decision variables."""
        start_time = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Dimension 5: [x_f, x_m, x_l, alt_ratio, speed]
        lb = np.array([0.0, 0.0, 0.0, 0.0, 9.5], dtype=float)
        ub = np.array([
            float(VESSEL_SPECS["feeder"]["max_available"]),
            float(VESSEL_SPECS["medium"]["max_available"]),
            float(VESSEL_SPECS["large"]["max_available"]),
            scenario.max_transition_rate,
            17.5,
        ], dtype=float)
        dim = len(lb)

        v_max = 0.20 * (ub - lb)
        v_min = -v_max

        # Initialize particles
        X = rng.uniform(lb, ub, size=(self.population_size, dim))
        V = rng.uniform(v_min, v_max, size=(self.population_size, dim))

        pbest = X.copy()
        pbest_scores = np.full(self.population_size, float("inf"))

        def decode_and_eval(vec: np.ndarray) -> tuple[float, dict[str, Any]]:
            xf = int(round(vec[0]))
            xm = int(round(vec[1]))
            xl = int(round(vec[2]))
            tot = xf + xm + xl
            if tot == 0:
                tot = 1
                xm = 1

            alt_ratio = float(np.clip(vec[3], 0.0, scenario.max_transition_rate))
            alt_count = int(round(tot * alt_ratio))
            n_lng = alt_count // 2
            n_meth = alt_count - n_lng
            n_diesel = tot - alt_count

            fuel_mix = {
                "diesel": max(0, n_diesel),
                "lng": max(0, n_lng),
                "methanol": max(0, n_meth),
                "hydrogen": 0,
                "ammonia": 0,
            }
            speed = float(np.clip(vec[4], 9.5, 17.5))
            res = self.evaluate_candidate(xf, xm, xl, fuel_mix, speed, scenario)
            return res["objective_score"], res

        best_eval_dict = None
        gbest = X[0].copy()
        gbest_score = float("inf")
        history: list[float] = []

        # Initial evaluation
        for i in range(self.population_size):
            score, res = decode_and_eval(X[i])
            pbest_scores[i] = score
            if score < gbest_score:
                gbest_score = score
                gbest = X[i].copy()
                best_eval_dict = res

        history.append(gbest_score)

        # Swarm iterations
        w_start, w_end = 0.9, 0.4
        c1, c2 = 1.5, 1.5

        for t in range(1, max_iterations):
            w = w_start - (t / max_iterations) * (w_start - w_end)

            r1 = rng.uniform(0.0, 1.0, size=(self.population_size, dim))
            r2 = rng.uniform(0.0, 1.0, size=(self.population_size, dim))

            V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
            V = np.clip(V, v_min, v_max)
            X = np.clip(X + V, lb, ub)

            for i in range(self.population_size):
                score, res = decode_and_eval(X[i])
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = X[i].copy()
                    if score < gbest_score:
                        gbest_score = score
                        gbest = X[i].copy()
                        best_eval_dict = res

            history.append(gbest_score)

        elapsed = time.perf_counter() - start_time
        assert best_eval_dict is not None

        return BenchmarkResult(
            solver_name=self.solver_name,
            runtime_seconds=round(elapsed, 4),
            iterations=max_iterations,
            objective_score=best_eval_dict["objective_score"],
            fuel_consumption=best_eval_dict["fuel_consumption"],
            operational_cost=best_eval_dict["operational_cost"],
            emissions=best_eval_dict["emissions"],
            reliability_score=best_eval_dict["reliability_score"],
            demand_satisfaction_rate=best_eval_dict["demand_satisfaction_rate"],
            convergence_score=round(float(history[0] - history[-1]) / max(abs(history[0]), 1e-4), 4),
            feasible_solution=best_eval_dict["feasible_solution"],
            metadata={
                "fleet_mix": best_eval_dict["fleet_mix"],
                "speed_knots": best_eval_dict["speed_knots"],
                "history": history,
            },
        )
