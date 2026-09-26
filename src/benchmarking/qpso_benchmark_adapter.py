"""Quantum Particle Swarm Optimization (QPSO) benchmark adapter.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Wraps the quantum delta-potential well QPSO algorithm into the common
BaseBenchmarkSolver interface for direct apples-to-apples comparison against classical baselines.
"""

import time
from typing import Any

import numpy as np

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver, VESSEL_SPECS


class QPSOBenchmarkAdapter(BaseBenchmarkSolver):
    """QPSO solver adapter conforming strictly to the benchmark evaluation interface."""

    def __init__(self, population_size: int = 20, alpha_start: float = 1.0, alpha_end: float = 0.5) -> None:
        super().__init__(solver_name="Quantum-Inspired PSO (QPSO)")
        self.population_size = population_size
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Execute QPSO search using quantum delta-potential-well dynamics."""
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
        dim = len(lb)

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
            if len(vec) >= 9:
                shares = [float(vec[4]), float(vec[5]), float(vec[6]), float(vec[7])]
                speed = float(np.clip(vec[8], 9.5, 17.5))
            else:
                shares = None
                speed = float(np.clip(vec[4], 9.5, 17.5))
            fuel_mix = self.allocate_fuel_mix(tot, alt_count, scenario, shares=shares)
            res = self.evaluate_candidate(xf, xm, xl, fuel_mix, speed, scenario)
            return res["objective_score"], res

        # 1. Initialize swarm positions uniformly in parameter bounds
        X = rng.uniform(lb, ub, size=(self.population_size, dim))
        pbest = X.copy()
        pbest_scores = np.full(self.population_size, float("inf"))
        eval_dicts: list[dict[str, Any]] = [{}] * self.population_size

        for i in range(self.population_size):
            pbest_scores[i], eval_dicts[i] = decode_and_eval(X[i])

        best_idx = int(np.argmin(pbest_scores))
        gbest = pbest[best_idx].copy()
        gbest_score = float(pbest_scores[best_idx])
        best_eval_dict = eval_dicts[best_idx]
        history: list[float] = [gbest_score]

        # 2. Quantum delta-potential updates
        for t in range(1, max_iterations):
            alpha = self.alpha_start - (t / max_iterations) * (self.alpha_start - self.alpha_end)
            mbest = np.mean(pbest, axis=0)

            for i in range(self.population_size):
                phi = rng.uniform(0.0, 1.0, size=dim)
                p = phi * pbest[i] + (1.0 - phi) * gbest

                u = rng.uniform(0.0, 1.0, size=dim)
                u = np.clip(u, 1e-12, 1.0)
                signs = rng.choice([-1.0, 1.0], size=dim)

                step = signs * alpha * np.abs(mbest - X[i]) * np.log(1.0 / u)
                X[i] = np.clip(p + step, lb, ub)

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

        conv_info = {
            "initial_objective": round(float(history[0]), 4),
            "final_objective": round(float(history[-1]), 4),
            "improvement_pct": round(float(history[0] - history[-1]) / max(abs(history[0]), 1e-4) * 100.0, 2),
            "history": history,
            "n_evaluations": eval_count,
        }

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
            n_evaluations=eval_count,
            convergence_information=conv_info,
            metadata={
                "fleet_mix": best_eval_dict["fleet_mix"],
                "speed_knots": best_eval_dict["speed_knots"],
                "history": history,
                "is_quantum": True,
            },
        )
