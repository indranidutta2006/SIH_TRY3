"""Convergence analysis and iteration trajectory tracking.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Tracks iteration-by-iteration objective descent, calculates improvement rates,
and quantifies quantum convergence speedup relative to classical evolutionary and swarm methods.
"""

import time
from typing import Any

from contracts.schemas import ConvergenceAnalysisResult, OptimizationScenario
from src.benchmarking.qpso_benchmark_adapter import QPSOBenchmarkAdapter
from src.benchmarking.solvers.classical_pso_solver import ClassicalPSOSolver
from src.benchmarking.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver
from src.benchmarking.solvers.simulated_annealing_solver import SimulatedAnnealingSolver


class ConvergenceAnalyzer:
    """Orchestrates convergence trace analysis across iterative optimization algorithms."""

    def __init__(self) -> None:
        """Initialize convergence analyzer with benchmark candidate solvers."""
        self.solvers = {
            "QPSO": QPSOBenchmarkAdapter(),
            "Classical PSO": ClassicalPSOSolver(),
            "Genetic Algorithm": GeneticAlgorithmSolver(),
            "Simulated Annealing": SimulatedAnnealingSolver(),
        }

    def run_convergence_suite(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> dict[str, ConvergenceAnalysisResult]:
        """Execute all iterative solvers on the identical scenario, capturing full convergence trajectories.

        Args:
            scenario: Comprehensive OptimizationScenario context.
            max_iterations: Number of optimization iterations.
            seed: Reproducibility random seed.

        Returns:
            Dictionary mapping solver names to ConvergenceAnalysisResult instances.
        """
        results: dict[str, ConvergenceAnalysisResult] = {}

        for name, solver in self.solvers.items():
            t0 = time.perf_counter()
            bench_res = solver.solve(scenario, max_iterations=max_iterations, seed=seed)
            total_time = time.perf_counter() - t0

            history = bench_res.metadata.get("history", [bench_res.objective_score])
            iterations = tuple(range(1, len(history) + 1))
            best_values = tuple(history)

            # Calculate relative improvement rate per iteration: (J_prev - J_curr) / J_prev
            improvement_rates: list[float] = [0.0]
            for i in range(1, len(history)):
                prev = history[i - 1]
                curr = history[i]
                rate = max(0.0, (prev - curr) / max(abs(prev), 1e-4))
                improvement_rates.append(round(rate, 6))

            # Detect convergence iteration: first iteration within 1% of final value
            final_obj = history[-1]
            conv_iter = len(history)
            for idx, val in enumerate(history):
                if abs(val - final_obj) <= 0.01 * max(abs(final_obj), 1e-4):
                    conv_iter = idx + 1
                    break

            results[name] = ConvergenceAnalysisResult(
                solver_name=name,
                iterations=iterations,
                objective_values=best_values,
                best_values=best_values,
                improvement_rates=tuple(improvement_rates),
                total_runtime_seconds=round(total_time, 4),
                final_objective=round(final_obj, 4),
                convergence_iteration=conv_iter,
                metadata={
                    "initial_objective": round(history[0], 4),
                    "total_improvement_pct": round(((history[0] - final_obj) / max(abs(history[0]), 1e-4)) * 100.0, 2),
                    "feasible": bench_res.feasible_solution,
                },
            )

        # Identify best-known objective across all solvers in this suite
        feasible_finals = [
            r.final_objective for r in results.values()
            if r.metadata.get("feasible", False)
        ]
        if feasible_finals:
            best_known_obj = min(feasible_finals)
        else:
            best_known_obj = min(r.final_objective for r in results.values())

        # Update each result with best_known metrics
        updated_results: dict[str, ConvergenceAnalysisResult] = {}
        for name, r in results.items():
            gap_pct = max(0.0, ((r.final_objective - best_known_obj) / max(abs(best_known_obj), 1e-4)) * 100.0)
            iter_2pct = len(r.objective_values)
            for idx, val in enumerate(r.objective_values):
                if (val - best_known_obj) / max(abs(best_known_obj), 1e-4) <= 0.02:
                    iter_2pct = idx + 1
                    break

            updated_results[name] = ConvergenceAnalysisResult(
                solver_name=r.solver_name,
                iterations=r.iterations,
                objective_values=r.objective_values,
                best_values=r.best_values,
                improvement_rates=r.improvement_rates,
                total_runtime_seconds=r.total_runtime_seconds,
                final_objective=r.final_objective,
                convergence_iteration=r.convergence_iteration,
                best_known_objective=round(best_known_obj, 4),
                gap_to_best_known_percent=round(gap_pct, 4),
                iterations_to_2pct_best_known=iter_2pct,
                metadata=r.metadata,
            )

        return updated_results

    def compute_quantum_speedup(
        self,
        convergence_results: dict[str, ConvergenceAnalysisResult],
    ) -> dict[str, float]:
        """Compute speedup and solution quality deltas of QPSO against classical methods."""
        qpso_res = convergence_results.get("QPSO")
        if qpso_res is None:
            return {}

        speedups: dict[str, float] = {}
        for name, res in convergence_results.items():
            if name == "QPSO":
                continue
            # Iteration speedup: how many fewer iterations QPSO needed to converge
            iter_ratio = res.convergence_iteration / max(qpso_res.convergence_iteration, 1)
            # Objective advantage: % better objective reached
            obj_adv = ((res.final_objective - qpso_res.final_objective) / max(abs(res.final_objective), 1e-4)) * 100.0

            speedups[f"{name}_iteration_speedup_factor"] = round(iter_ratio, 2)
            speedups[f"{name}_objective_improvement_pct"] = round(obj_adv, 2)

        return speedups
