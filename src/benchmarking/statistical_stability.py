"""Statistical stability and Monte Carlo repeatability testing.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Executes stochastic optimization solvers across multiple independent random seeds
(N=30 standard) to demonstrate statistical consistency and eliminate stochastic luck claims.
"""

import time
from typing import Any

import numpy as np

from contracts.schemas import OptimizationScenario, StatisticalStabilityResult
from src.benchmarking.qpso_benchmark_adapter import QPSOBenchmarkAdapter
from src.benchmarking.solvers.classical_pso_solver import ClassicalPSOSolver
from src.benchmarking.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver
from src.benchmarking.solvers.simulated_annealing_solver import SimulatedAnnealingSolver


class StatisticalStabilityEvaluator:
    """Evaluates multi-seed stochastic stability across optimization solvers."""

    def __init__(self) -> None:
        """Initialize stability evaluator with stochastic benchmark solvers."""
        self.solvers = {
            "QPSO": QPSOBenchmarkAdapter(),
            "Classical PSO": ClassicalPSOSolver(),
            "Genetic Algorithm": GeneticAlgorithmSolver(),
            "Simulated Annealing": SimulatedAnnealingSolver(),
        }

    def evaluate_solver_stability(
        self,
        solver_name: str,
        scenario: OptimizationScenario,
        num_seeds: int = 30,
        base_seed: int = 100,
        max_iterations: int = 25,
    ) -> StatisticalStabilityResult:
        """Execute a single solver across multiple seeds and compute distribution metrics."""
        solver = self.solvers.get(solver_name)
        if solver is None:
            raise ValueError(f"Unknown stochastic solver: {solver_name}")

        seeds = tuple(base_seed + i for i in range(num_seeds))
        objective_values: list[float] = []
        feasible_count = 0

        for s in seeds:
            res = solver.solve(scenario, max_iterations=max_iterations, seed=s)
            objective_values.append(res.objective_score)
            if res.feasible_solution:
                feasible_count += 1

        obj_arr = np.array(objective_values, dtype=float)
        mean_val = float(np.mean(obj_arr))
        std_val = float(np.std(obj_arr))
        best_val = float(np.min(obj_arr))
        worst_val = float(np.max(obj_arr))

        return StatisticalStabilityResult(
            solver_name=solver_name,
            num_seeds=num_seeds,
            seeds=seeds,
            objective_values=tuple(round(v, 4) for v in objective_values),
            mean_objective=round(mean_val, 4),
            std_objective=round(std_val, 4),
            best_objective=round(best_val, 4),
            worst_objective=round(worst_val, 4),
            metadata={
                "coefficient_of_variation": round((std_val / max(abs(mean_val), 1e-4)) * 100.0, 2),
                "iqr": round(float(np.percentile(obj_arr, 75) - np.percentile(obj_arr, 25)), 4),
                "feasible_run_count": feasible_count,
                "feasible_run_rate": round(feasible_count / max(1, num_seeds), 4),
            },
        )

    def evaluate_all_stochastic_solvers(
        self,
        scenario: OptimizationScenario,
        num_seeds: int = 30,
        base_seed: int = 100,
        max_iterations: int = 25,
    ) -> dict[str, StatisticalStabilityResult]:
        """Evaluate statistical stability across all stochastic benchmark solvers."""
        results: dict[str, StatisticalStabilityResult] = {}
        for name in self.solvers:
            results[name] = self.evaluate_solver_stability(
                solver_name=name,
                scenario=scenario,
                num_seeds=num_seeds,
                base_seed=base_seed,
                max_iterations=max_iterations,
            )
        return results
