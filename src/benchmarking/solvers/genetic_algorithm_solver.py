"""Genetic Algorithm (GA) benchmark solver.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Implements evolutionary algorithm with tournament selection,
arithmetic crossover, mutation, and elitism for maritime fleet optimization.
"""

import time
from typing import Any

import numpy as np

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver, VESSEL_SPECS


class GeneticAlgorithmSolver(BaseBenchmarkSolver):
    """Genetic Algorithm baseline solver."""

    def __init__(self, population_size: int = 24, mutation_rate: float = 0.15) -> None:
        super().__init__(solver_name="Genetic Algorithm")
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Execute Genetic Algorithm search over fleet decisions."""
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

        # Initialize population
        pop = rng.uniform(lb, ub, size=(self.population_size, dim))
        scores = np.empty(self.population_size, dtype=float)
        eval_dicts: list[dict[str, Any]] = [{}] * self.population_size

        for i in range(self.population_size):
            scores[i], eval_dicts[i] = decode_and_eval(pop[i])

        best_idx = int(np.argmin(scores))
        gbest = pop[best_idx].copy()
        gbest_score = float(scores[best_idx])
        best_eval_dict = eval_dicts[best_idx]
        history: list[float] = [gbest_score]

        # Generations
        for _ in range(1, max_iterations):
            new_pop = np.empty_like(pop)

            # 1. Elitism: preserve top 2
            sorted_indices = np.argsort(scores)
            new_pop[0] = pop[sorted_indices[0]].copy()
            new_pop[1] = pop[sorted_indices[1]].copy()

            # 2. Reproduction
            for j in range(2, self.population_size, 2):
                # Tournament selection
                i1, i2 = rng.choice(self.population_size, size=2, replace=False)
                p1 = pop[i1] if scores[i1] < scores[i2] else pop[i2]

                i3, i4 = rng.choice(self.population_size, size=2, replace=False)
                p2 = pop[i3] if scores[i3] < scores[i4] else pop[i4]

                # Arithmetic Crossover
                alpha = rng.uniform(0.1, 0.9, size=dim)
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = (1.0 - alpha) * p1 + alpha * p2

                # Mutation
                for child in (c1, c2):
                    if rng.uniform() < self.mutation_rate:
                        mutation_step = rng.normal(0.0, 0.1 * (ub - lb))
                        child += mutation_step

                new_pop[j] = np.clip(c1, lb, ub)
                if j + 1 < self.population_size:
                    new_pop[j + 1] = np.clip(c2, lb, ub)

            pop = new_pop
            for i in range(self.population_size):
                scores[i], eval_dicts[i] = decode_and_eval(pop[i])
                if scores[i] < gbest_score:
                    gbest_score = scores[i]
                    gbest = pop[i].copy()
                    best_eval_dict = eval_dicts[i]

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
            },
        )
